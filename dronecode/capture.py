#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
capture.py: Triggering image captures based on RC MAVLink messages

Each captured image gets a matching .json file next to it holding the drone
telemetry as it was at the moment of capture.

"""

###############################################
# Definitions                                 #
###############################################
RC_CAPTURE = 6 # RX Channel 7

# What asked for an image. Ends up in the "trigger" field of the JSON file.
TRIGGER_SWITCH = "switch"          # the RC switch on channel RC_CAPTURE
TRIGGER_QGC = "qgroundcontrol"     # a photo event on a mission waypoint

# Exposure min/max supported for Cam v2: 75, 11766829
# Daytime aerial values: 250-500 full sun, 1000 hazy sun, 2000-4000 overcast.
# Longer than ~2000 starts to smear at normal survey speeds and altitudes.
CAM_EXPOSURE_TIME = 1100 #in micro seconds, 1000 microsecond => 1ms
CAM_GAIN = 1.0 # analogue gain, 1.0 is the least noisy
CAM_AE = False # Auto Exposure setting. If this is enabled the manual exposure time and gain will be overridden
CAM_AWB = True # Auto white balance, in case you do not want to let it auto adjust the white balance

# Optics of the Pi Camera v2 (IMX219), used to estimate ground resolution and
# motion blur for each image. Change these if you fit a different camera.
CAM_PIXEL_PITCH_M = 1.12e-6   # sensor pixel size in metres
CAM_FOCAL_LENGTH_M = 3.04e-3  # lens focal length in metres
# Angle covered by one pixel (radians). Ground resolution is this times the
# height above ground.
CAM_IFOV_RAD = CAM_PIXEL_PITCH_M / CAM_FOCAL_LENGTH_M

# Telemetry we send back out, so the operator can see how many images we have
# taken without looking at the Pi. Shows up in QGroundControl under
# Analyze > MAVLink Inspector as NAMED_VALUE_INT, and can be plotted.
# The MAVLink name field holds at most 10 characters.
IMG_COUNT_NAME = "IMG_COUNT"
IMG_COUNT_INTERVAL_S = 2.0  # how often we re-send the count when idle

# MAVLink messages we subscribe to for the image telemetry, and the rate (Hz)
# we ask the autopilot to send them at.
TELEMETRY_STREAMS = {
    'GLOBAL_POSITION_INT': 5,   # fused position, altitude, ground speed, heading
    'GPS_RAW_INT': 2,           # raw GPS: fix type, satellites, accuracy
    'ATTITUDE': 10,             # roll/pitch/yaw of the airframe
    'VFR_HUD': 5,               # airspeed, groundspeed, climb rate, throttle
    'SYS_STATUS': 1,            # battery voltage/current/remaining
}

###############################################
# Standard Imports                            #
###############################################
import time
import threading
import os
import json
import math

###############################################
# MAVlink Imports                             #
###############################################
from pymavlink import mavutil


###############################################
# OpenCV Imports                              #
###############################################


###############################################
# RPi Imports                                 #
###############################################
import picamera2
import datetime

###############################################
# Drone Control class                         #
###############################################
class DroneControl:
    def __init__(self, *args):
        self.ns = "DroneControl"
        self.armed = False
        self.mode = None
        self.rc_channels = [0,0,0,0,0,0,0,0,0,0,0,0]
        self.index = 0
        self.debounce = False
        self.uav_position = -1
        self.debug = True
        self.first_image_taken = False

        self.state = "INIT"
        self.gps_loc_filename='gps_locations.log'

        # Used for the time_boot_ms field of the telemetry we send out, and to
        # pace how often we re-send the image count.
        self.boot_time = time.monotonic()
        self.last_img_count_sent = 0.0

        # Latest telemetry, keyed by message group. Written by the MAVLink
        # thread, read by the capture thread, so guard it with a lock.
        self.telemetry = {}
        self.telemetry_lock = threading.Lock()

        # Capture requests coming in over MAVLink (mission photo events). The
        # MAVLink thread appends to this, the main loop takes them off and does
        # the actual capture, so only one thread ever touches the camera.
        self.pending_triggers = []
        self.trigger_lock = threading.Lock()

        # Both threads send on the MAVLink link (the main loop sends the image
        # count, the MAVLink thread sends heartbeats and stream requests when it
        # reconnects), so serialise the sends. Only ever held around the send
        # itself, never around anything that blocks waiting for the link.
        self.mav_send_lock = threading.Lock()

        # Make New folder for photos


        self.camera = self.init_camera()  # change camera properties within this function

        # initialise MAVLink Connection

        self.interface = mavutil.mavlink_connection("/dev/PX4", baud=115200, autoreconnect=True)

        with self.mav_send_lock:
            self.interface.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        self.interface.wait_heartbeat()
        self.request_telemetry()

        self.t_run = threading.Thread(target=self.recv_mavlink)
        self.set_state("RUNNING")
        self.t_run.start()
        self.print_debug(">> MAVlink communication is running (Thread)")
        ## end __init__

    def init_camera(self):
        '''
        Initialise the raspberry pi camera with desired properties. 
        For information on the picamera API, see following URL:
        https://picamera.readthedocs.io/en/release-1.13/api_camera.html#
        '''
        camera = picamera2.Picamera2()

        config = camera.create_still_configuration()
        camera.configure(config)
        camera.set_controls({"ExposureTime": CAM_EXPOSURE_TIME, "AnalogueGain": CAM_GAIN, "AeEnable": CAM_AE, "AwbEnable": CAM_AWB})
        camera.start()
        
        return camera

    def print_debug(self,msg):
        ''' Print a message to the console with the prefix "[DroneControl]: '''
        if self.debug == True:
            print("[{}]: >> {} ".format(self.ns,msg))

    def request_telemetry(self):
        '''
        Ask the autopilot to stream the MAVLink messages listed in
        TELEMETRY_STREAMS. Most of them are already streamed by default, this
        just makes sure they arrive often enough to be useful per image.
        '''
        for name, rate_hz in TELEMETRY_STREAMS.items():
            msg_id = getattr(mavutil.mavlink, 'MAVLINK_MSG_ID_' + name, None)
            if msg_id is None:
                self.print_debug('WARN: unknown MAVLink message {}'.format(name))
                continue
            try:
                with self.mav_send_lock:
                    self.interface.mav.command_long_send(
                        self.interface.target_system,
                        self.interface.target_component,
                        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                        0,                          # confirmation
                        msg_id,                     # param1: message id
                        int(1e6 / rate_hz),         # param2: interval in microseconds
                        0, 0, 0, 0, 0)
            except Exception as e:
                self.print_debug('WARN: could not request {}: {}'.format(name, e))

    def store_telemetry(self, group, data):
        ''' Store one group of telemetry values together with its arrival time. '''
        data['received_at'] = time.monotonic()
        with self.telemetry_lock:
            self.telemetry[group] = data

    def snapshot_telemetry(self):
        '''
        Take a copy of the latest telemetry and replace the arrival timestamp
        with the age of the data (in seconds) at the time of the snapshot, so a
        stale value is easy to spot when looking at the JSON file afterwards.
        '''
        now = time.monotonic()
        with self.telemetry_lock:
            snapshot = {group: dict(values) for group, values in self.telemetry.items()}

        for values in snapshot.values():
            values['age_s'] = round(now - values.pop('received_at'), 3)
        return snapshot

    def send_image_count(self, message=None):
        '''
        Tell the operator how many images we have taken. The count goes out as
        NAMED_VALUE_INT so it can be watched or plotted in QGroundControl, and
        an optional message goes out as STATUSTEXT so it also shows up in the
        message panel. Sending is best effort: if the link is down we carry on
        capturing rather than dying here.
        '''
        self.last_img_count_sent = time.monotonic()
        time_boot_ms = int((self.last_img_count_sent - self.boot_time) * 1e3)
        try:
            with self.mav_send_lock:
                self.interface.mav.named_value_int_send(
                    time_boot_ms, IMG_COUNT_NAME.encode('ascii'), self.index)
                if message:
                    self.interface.mav.statustext_send(
                        mavutil.mavlink.MAV_SEVERITY_INFO,
                        message.encode('ascii')[:50])
        except Exception as e:
            self.print_debug('WARN: could not send image count: {}'.format(e))

    def set_state(self, data):
        self.state = data
        self.print_debug("New State: {}".format(data))

    def request_capture(self, trigger):
        ''' Queue a capture asked for over MAVLink. Called from the MAVLink thread. '''
        with self.trigger_lock:
            self.pending_triggers.append(trigger)

    def take_pending_trigger(self):
        ''' Take the oldest queued capture request, or None if there is none. '''
        with self.trigger_lock:
            if self.pending_triggers:
                return self.pending_triggers.pop(0)
        return None

    def capture(self, trigger):
        '''
        Captures an image plus its telemetry. "trigger" records what asked for
        the image: TRIGGER_SWITCH for the RC switch, TRIGGER_QGC for a photo
        event on a mission waypoint.
        '''

        if(not self.first_image_taken):
            self.create_folder()
            self.first_image_taken = True

        img_loc = self.folder_loc+'img_'+str(self.index)

        # Take the next frame off the already-running camera. Unlike
        # start_and_capture_file this does not stop/restart the camera or wait
        # a settle delay, so the frame we get was exposed within the last frame
        # period, and the request carries the real sensor metadata for it.
        request = self.camera.capture_request()

        # sample the telemetry while the shutter is still warm: we are now
        # within ~0.1 s of the exposure instead of the ~1.5 s the old
        # stop/settle/restart cycle put between snapshot and shutter
        telemetry = self.snapshot_telemetry()
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        metadata = request.get_metadata()

        request.save('main', img_loc+'.jpg')
        request.release()
        msg="Image captured ({}). Saved image as img_{}.jpg".format(trigger, self.index)
        self.print_debug(msg)

        # save the full telemetry next to the image
        self.write_telemetry_json(img_loc, telemetry, timestamp, trigger, metadata)

        # save geolocation of image
        try:
            write_drone_location = open(self.folder_loc+self.gps_loc_filename,'a')

            output = '{},{:.7f},{:.7f},{:.2f},{}\n'.format(
                self.index,
                self.uav_position[0],
                self.uav_position[1],
                self.uav_position[2],
                self.uav_position[3],
            )

            write_drone_location.write(output)
            write_drone_location.close()
        except TypeError:
            self.print_debug('WARN: No GPS fix... skipping file save')
            pass
        self.index += 1

        # let the operator know an image went in the can
        self.send_image_count('Image {} captured ({})'.format(self.index, trigger))

    def estimate_image_quality(self, telemetry, exposure_us=CAM_EXPOSURE_TIME):
        '''
        Estimate ground resolution and motion blur for one image from the
        telemetry taken at the shutter. Blur is given in pixels: below 1 the
        image is sharp, above ~2 it is visibly smeared. Returns None for
        anything that cannot be worked out from the telemetry we have.
        '''
        position = telemetry.get('position', {})
        attitude = telemetry.get('attitude', {})
        exposure_s = exposure_us / 1e6

        quality = {'exposure_s': exposure_s}

        # Height above the takeoff point, the best stand-in for height above
        # ground we have without a rangefinder.
        altitude = position.get('alt_relative_m')
        gsd = CAM_IFOV_RAD * altitude if altitude is not None else None
        quality['gsd_m_per_px'] = round(gsd, 4) if gsd else None

        # Blur from flying forwards: how far the ground moves under the camera
        # while the shutter is open, in pixels.
        if gsd and 'vx_ms' in position:
            groundspeed = math.hypot(position['vx_ms'], position['vy_ms'])
            quality['groundspeed_ms'] = round(groundspeed, 2)
            quality['motion_blur_px'] = round(groundspeed * exposure_s / gsd, 2)

        # Blur from turning. This does not depend on altitude, and at survey
        # speeds it is usually the larger of the two.
        if 'yawspeed_dps' in attitude:
            rate = math.hypot(attitude['rollspeed_dps'], attitude['pitchspeed_dps'],
                              attitude['yawspeed_dps'])
            quality['rotation_rate_dps'] = round(rate, 2)
            quality['rotation_blur_px'] = round(
                math.radians(rate) * exposure_s / CAM_IFOV_RAD, 2)

        return quality

    def write_telemetry_json(self, img_loc, telemetry, timestamp, trigger, metadata=None):
        '''
        Write img_<n>.json next to img_<n>.jpg, holding the telemetry as it was
        when the image was taken. "metadata" is the picamera2 request metadata
        of the saved frame; it holds what the sensor actually did, which is not
        the requested settings when auto exposure is on.
        '''
        metadata = metadata or {}
        exposure_us = metadata.get('ExposureTime', CAM_EXPOSURE_TIME)
        record = {
            'image': os.path.basename(img_loc) + '.jpg',
            'index': self.index,
            'timestamp_utc': timestamp.isoformat(),
            'trigger': trigger,
            'armed': self.armed,
            'mode': self.mode,
            'camera': {
                'exposure_time_us': exposure_us,
                'analogue_gain': metadata.get('AnalogueGain', CAM_GAIN),
                'auto_exposure': CAM_AE,
                'auto_white_balance': CAM_AWB,
            },
            'telemetry': telemetry,
            'quality': self.estimate_image_quality(telemetry, exposure_us),
        }

        try:
            with open(img_loc + '.json', 'w') as f:
                json.dump(record, f, indent=2)
        except OSError as e:
            self.print_debug('WARN: could not write telemetry file: {}'.format(e))

    def create_folder(self):
        count = 0
        self.folder_loc = '/home/pi/images/capture_{}/'.format(count)
        
        if not os.path.exists(self.folder_loc):
            os.mkdir(self.folder_loc)
        else:
            while(os.path.exists(self.folder_loc)):
                count+=1
                self.folder_loc = '/home/pi/images/capture_{}/'.format(count)
            os.mkdir(self.folder_loc)
                    
        self.print_debug('Saving media to: {}'.format(self.folder_loc))

    def recv_mavlink(self):
        while self.state == "RUNNING":
            try:
                m = self.interface.recv_msg()
            except Exception as e:
                self.debug = False
                print("Lost Connection. Device has possibly been rebooted. Trying to reconnect ...")
                try:
                    with self.mav_send_lock:
                        # swap the link and greet it under the lock, so the main
                        # loop cannot send on a half-replaced connection
                        self.interface = mavutil.mavlink_connection("/dev/PX4", baud=115200, autoreconnect=True)
                        self.interface.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                    self.interface.wait_heartbeat()
                    self.request_telemetry()
                except:
                    pass
                time.sleep(1)


            #print(m)
            if m:
                self.debug = True
                #print(m)
                if(m.get_type() == 'HEARTBEAT'):
                    #print(m)
                    self.armed = self.decode_armed(m.base_mode)
                    self.mode = self.decode_custom_flightmode(m.custom_mode)
                elif(m.get_type() == 'RC_CHANNELS'):
                    #self.print_debug(m.chan1_raw)
                    self.rc_channels = [m.chan1_raw, # Throttle
                                     m.chan2_raw, # Roll
                                     m.chan3_raw, # Pitch
                                     m.chan4_raw, # Yaw
                                     m.chan5_raw,
                                     m.chan6_raw,
                                     m.chan7_raw,
                                     m.chan8_raw,
                                     m.chan9_raw,
                                     m.chan10_raw,
                                     m.chan11_raw,
                                     m.chan12_raw]
                elif (m.get_type() == 'GLOBAL_POSITION_INT'):
                    # print(m.lat/1e7, m.lon/1e7, m.alt/1e3)
                    self.uav_position = (
                        m.lat/1e7, m.lon/1e7, m.alt/1e3, int(m.hdg/1e2))
                    #self.print_debug(self.rc_channels)
                    self.store_telemetry('position', {
                        'lat_deg': m.lat/1e7,
                        'lon_deg': m.lon/1e7,
                        'alt_amsl_m': m.alt/1e3,       # above mean sea level
                        'alt_relative_m': m.relative_alt/1e3,  # above takeoff point
                        'vx_ms': m.vx/1e2,             # north
                        'vy_ms': m.vy/1e2,             # east
                        'vz_ms': m.vz/1e2,             # down
                        'heading_deg': m.hdg/1e2,
                    })
                elif (m.get_type() == 'GPS_RAW_INT'):
                    self.store_telemetry('gps', {
                        'fix_type': m.fix_type,        # 3 = 3D fix, see GPS_FIX_TYPE
                        'satellites_visible': m.satellites_visible,
                        'eph': m.eph/1e2,              # horizontal dilution of precision
                        'epv': m.epv/1e2,              # vertical dilution of precision
                        'alt_amsl_m': m.alt/1e3,
                    })
                elif (m.get_type() == 'ATTITUDE'):
                    self.store_telemetry('attitude', {
                        'roll_deg': math.degrees(m.roll),
                        'pitch_deg': math.degrees(m.pitch),
                        'yaw_deg': math.degrees(m.yaw),
                        'rollspeed_dps': math.degrees(m.rollspeed),
                        'pitchspeed_dps': math.degrees(m.pitchspeed),
                        'yawspeed_dps': math.degrees(m.yawspeed),
                    })
                elif (m.get_type() == 'VFR_HUD'):
                    self.store_telemetry('hud', {
                        'airspeed_ms': m.airspeed,
                        'groundspeed_ms': m.groundspeed,
                        'heading_deg': m.heading,
                        'throttle_percent': m.throttle,
                        'alt_amsl_m': m.alt,
                        'climb_ms': m.climb,
                    })
                elif (m.get_type() == 'CAMERA_TRIGGER'):
                    # PX4 emits this when a mission camera trigger fires, i.e. a
                    # photo event placed on a waypoint in QGroundControl.
                    self.store_telemetry('camera_trigger', {'seq': m.seq})
                    self.request_capture(TRIGGER_QGC)
                elif (m.get_type() == 'COMMAND_LONG'):
                    # Fallback for setups where the trigger command reaches us
                    # directly instead of PX4 turning it into a CAMERA_TRIGGER.
                    if m.command in (mavutil.mavlink.MAV_CMD_IMAGE_START_CAPTURE,
                                     mavutil.mavlink.MAV_CMD_DO_DIGICAM_CONTROL):
                        self.request_capture(TRIGGER_QGC)
                elif (m.get_type() == 'SYS_STATUS'):
                    self.store_telemetry('battery', {
                        'voltage_v': m.voltage_battery/1e3,
                        # -1 means the autopilot does not measure it
                        'current_a': m.current_battery/1e2 if m.current_battery != -1 else None,
                        'remaining_percent': m.battery_remaining if m.battery_remaining != -1 else None,
                    })

        self.print_debug(">> MAVlink communication has stopped...")
        self.set_state("STOP")

    def decode_armed(self, base_mode):
        return bool((base_mode & 0x80) >> 7)

    def decode_custom_flightmode(self, custom_mode):
        # get bits for sub and main mode
        sub_mode = (custom_mode >> 24)
        main_mode = (custom_mode >> 16) & 0xFF

        # set default value to none
        main_mode_text = ""
        sub_mode_text = ""

        # make list of modes
        mainmodeList = ["MANUAL", "ALTITUDE CONTROL", "POSITION CONTROL", "AUTO", "ACRO", "OFFBOARD", "STABILIZED", "RATTITUDE"]
        submodeList = ["READY", "TAKEOFF", "LOITER", "MISSION", "RTL", "LAND", "RTGS", "FOLLOW_TARGET", "PRECLAND"]

        # get mode from list based on index and what mode we have active. -1 as values is 0-indexed
        # print(main_mode)
        main_mode_text = mainmodeList[main_mode - 1]
        sub_mode_text = submodeList[sub_mode - 1]

        # if no submode, just return mainmode. As a result of -1 will wrap around the array and give precland
        if sub_mode <= 0 or sub_mode > 9:
            return main_mode_text
        else:
            return main_mode_text + " - " + sub_mode_text
    
    def run(self):
        '''
        This is the main loop function
        Add your code in here.
        '''

        # forever loop
        try:
            while True:
                # photo events from the mission, queued by the MAVLink thread
                trigger = self.take_pending_trigger()
                if trigger:
                    self.capture(trigger)

                # the RC switch, captures once per flick of the switch
                if self.rc_channels[RC_CAPTURE] > 1500:
                    if not self.debounce:
                        self.capture(TRIGGER_SWITCH)
                        self.debounce = True
                elif self.debounce:
                    self.debounce = False

                # keep the count on the operator's screen even when nothing is
                # being captured, so it survives a GCS reconnect
                if time.monotonic() - self.last_img_count_sent > IMG_COUNT_INTERVAL_S:
                    self.send_image_count()

                self.print_debug("\033c Testing: \n Armed: {} \n Mode {} \n Images: {} \n RC Channels: {} \n".format(
                    self.armed, self.mode, self.index, self.rc_channels))
                time.sleep(0.1)
        

        # Kill MAVlink connection when Ctrl-C is pressed (results in a lock)
        except KeyboardInterrupt:
            self.print_debug('CTRL-C has been pressed! Exiting...')
            self.state = "EXIT"
            exit()


if __name__ == '__main__':
    DC = DroneControl()
    DC.print_debug('Brief pause before continuing....')
    time.sleep(1.0)
    DC.run()
