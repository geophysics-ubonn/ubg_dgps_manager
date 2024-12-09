# Uni-Bonn dGPS Manager

## Components of the dGPS Manager

* **Importer and 3D point manager**
* **Electrode Manager** (2D point manager): Manage 2D point locations derived from
  the 3D data points.
  Basically, x and z coordinates are used.
  The resulting coordinates can then be used for further processing, e.g., for
  2D seismic, 2D GPR, or 2D ERT data analysis.
* **ERT/IP mesh creator**
  Create meshes for electrical resistivity analysis

## Processing Cycle

Usually gps is data is repeatedly processed/analysed with different targets in mind.
For example, gps data from one day could contain different data layers that
contain gps points for multiple methods and/or profiles.
The analysis of gps data with regard to one target (i.e., one ERT profile) is
here called an analysis cycle.

* Each analysis cycle should be assigned a unique label id, e.g.
  *day2_ert_profile1*. This id is referred to as *[ID]*.

## Metadata Managing

* After preparing the GPS/2D coordinates, make sure to

	* Store the LOGs of both the **gps manager** and the **electrode manager**
	  in files called *gps_processing_[ID]_gps.txt* and
	  *gps_processing_[ID]_2d.txt*.
	  The log file ending in *_gps.txt* holds the log of the 3D point manager,
	  while the file ending in *_2d.txt* holds the log of the 2D electrode
	  manager.
	* Processed gps coordinates go into *.gps* files
	* Processed 2d coordinates go into .2d file

* Resulting directory layout that can then be used for data managing.
  An example listing, producing one point data set for an ert profile, and one
  for a seismics profile could look like this:

		RawData/
			gps_points_day2_GeoJSON.zip
		ProcessedData/
			day2_ert_profile1_processing_gps.txt
			day2_ert_profile1_processing_2d.txt
			day2_ert_profile1.gps
			day2_ert_profile1.2d
			day2_seismic_profile1_processing_gps.txt
			day2_seismic_profile1_processing_2d.txt
			day2_seismic_profile1.gps
			day2_seismic_profile1.2d
		ERT_Mesh/
			day2_ert_profile1_mesh.zip

## TODO

* Populate this help text
* 2D Coordinate export has y=1 cooridnates -> must be 0
* clear line log inputs after adding to log (gps and 2d managers)
* can we add functionality to download/copy-all/select-all logs?
* CRTomo mesh creator:
	* Better visual feedback while generating mesh
	* Can we print element count
