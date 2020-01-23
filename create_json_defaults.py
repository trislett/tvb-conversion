#!/usr/bin/env python

import os
import sys
import json
import argparse as ap

DESCRIPTION = "Defaults for TVB pipeline."

def getArgumentParser(parser = ap.ArgumentParser(description = DESCRIPTION)):
	parser.add_argument('--recon_all_dir',
		nargs = 1,
		metavar = ('str'),
		default = ['/mnt/mbServerData/data/FREESURFER_SUBJECTS/IntegraMent'],
		help='Path to the recon-all results. Default: %(default)s)')
	parser.add_argument('--dwi_output_directory',
		nargs = 1,
		metavar = ('str'),
		default = ['/mnt/mbServerProjects/SCRATCH/USERS/tris/CONNECTOME_INTEGRAMENT/DWI/dwitract'],
		help = 'Path to results of MRtrix3_connectome pipeline. Default: %(default)s)')
	parser.add_argument('--tvb_output',
		nargs = 1,
		metavar = ('str'),
		default = ['/mnt/mbServerProjects/SCRATCH/USERS/tris/CONNECTOME_INTEGRAMENT/TVB/TVB_OUTPUT'],
		help='Path to results of TVBconverter pipeline. Default: %(default)s)')
	parser.add_argument('-p', '--parcellation',
		nargs = 1,
		choices = ['hcpmmp1'],
		default = ['hcpmmp1'],
		help = 'Parcellation used in MRtrix3_connectome pipeline. {%(choices)s} Default: %(default)s)')
	parser.add_argument('--outputnames',
		nargs = 1,
		choices = ['default', 'script'],
		default = ['default'],
		help = 'The choice of output naming defautls for the zip files. {%(choices)s} Default: %(default)s)')
	return parser

def run(opts):

	processing_settings = {}

	# input standard image, its mask, and its headmask
	processing_settings['recon_all_dir'] = opts.recon_all_dir[0]
	processing_settings['dwi_output_directory'] = opts.dwi_output_directory[0]
	processing_settings['tvb_output'] = opts.tvb_output[0]

	processing_settings['parcellation'] = opts.parcellation[0]

	# default names
	processing_settings['default_file_names'] = {}
	processing_settings['default_file_names']['weights'] = "connectome.csv"
	processing_settings['default_file_names']['tracts'] = "meanlength.csv"
	if opts.parcellation[0] == 'hcpmmp1':
		processing_settings['default_file_names']['parc_image'] = "hcp_mmp1_parc_in_dwi.nii"

	# output names
	if opts.outputnames[0] == 'default':
		processing_settings['connectome_file_names'] = {}
		processing_settings['connectome_file_names']['weights'] = 'weights.txt'
		processing_settings['connectome_file_names']['tract_lengths'] = 'tract_lengths.txt'
		processing_settings['connectome_file_names']['hemispheres'] = 'hemispheres.txt'
		processing_settings['connectome_file_names']['centres'] = 'centres.txt'
		processing_settings['connectome_file_names']['average_orientations'] = 'average_orientations.txt'
		processing_settings['connectome_file_names']['areas'] = 'areas.txt'
		processing_settings['connectome_file_names']['cortical'] = 'cortical.txt'
	elif opts.outputnames[0] == 'script':
		processing_settings['connectome_file_names'] = {}
		processing_settings['connectome_file_names']['weights'] = 'weights.txt'
		processing_settings['connectome_file_names']['tract_lengths'] = 'tract.txt'
		processing_settings['connectome_file_names']['hemispheres'] = 'hemisphere.txt'
		processing_settings['connectome_file_names']['centres'] = 'centres.txt'
		processing_settings['connectome_file_names']['average_orientations'] = 'orientation.txt'
		processing_settings['connectome_file_names']['areas'] = 'area.txt'
		processing_settings['connectome_file_names']['cortical'] = 'cortical.txt'
	else:
		pass

	# names for conversion
	processing_settings['mrtrix_conversion'] = {}
	if opts.parcellation[0] == 'hcpmmp1':
		processing_settings['mrtrix_conversion']['label_info'] = os.path.dirname(os.path.realpath(__file__)) + "/HCP_label_info.txt"
		processing_settings['mrtrix_conversion']['borked_names'] = ['R_RO1', 'R_RO2', 'R_PSR', 'R_SFR', 'R_5R', 'R_7AR', 'R_7PR', 'R_RIPv', 'R_8BR', 'R_47R', 'R_11R', 'R_13R', 'R_RIPd', 'R_PBeRt', 'R_RO3', 'R_MBeRt', 'R_RBeRt']
		processing_settings['mrtrix_conversion']['fixed_names'] = ['R_LO1', 'R_LO2', 'R_PSL', 'R_SFL', 'R_5L', 'R_7AL', 'R_7PL', 'R_LIPv', 'R_8BL', 'R_47l', 'R_11l', 'R_13l', 'R_LIPd', 'R_PBelt', 'R_LO3', 'R_MBelt', 'R_LBelt']

	# Write out the processing settings
	with open('processing_settings.json', 'w') as outfile:
		json.dump(processing_settings, outfile, indent=3, sort_keys=True)

	# Load saved JSON and print the settings
	print("####\tPRINTING JSON FILE\t####")
	with open('processing_settings.json') as json_file:
		processing_settings = json.load(json_file)

	print(json.dumps(processing_settings, indent=3))

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
