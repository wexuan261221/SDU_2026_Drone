# Build the colour profiles and the observations map.
#
#   make               build everything that is out of date
#   make profiles      just the colour profiles
#   make clean         remove all generated files

PYTHON := uv run python

PROFILES := color_profiles/grass.json color_profiles/tarmac.json

.PHONY: all profiles clean

all: observations.png

profiles: $(PROFILES)

color_profiles/grass.json: $(wildcard color_profiles/grass/*) color_profiles/color_profile.py
	$(PYTHON) color_profiles/color_profile.py color_profiles/grass

color_profiles/tarmac.json: $(wildcard color_profiles/tarmac/*) color_profiles/color_profile.py
	$(PYTHON) color_profiles/color_profile.py color_profiles/tarmac

observations.png: $(PROFILES) analyze.py animalfinder.py $(wildcard capture_18/*)
	$(PYTHON) animalfinder.py capture_18 -o $@

clean:
	rm -f observations.png $(PROFILES)
	rm -rf __pycache__ color_profiles/__pycache__
