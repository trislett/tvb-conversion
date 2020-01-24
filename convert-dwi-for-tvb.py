#!/usr/bin/env python

import os
import json
import mne
import shutil
import glob

import numpy as np
import argparse as ap
import nibabel as nib
import scipy.io as sio

from mne.surface import _project_onto_surface
from mne.bem import _surfaces_to_bem, write_bem_surfaces
from mne.io.constants import FIFF
from scipy import spatial
from subprocess import Popen, PIPE

from functions import parc_get_mean_coord, subset_surface_by_cortex_label, create_adjac_vertex, vectorized_surface_smooth, get_surface_area, write_out_labels, compute_normals, eeg_sensors_to_surface, write_surf_tvb


DESCRIPTION = "Convert data of MRtrix3_connectome and fmriprep pipelines to TVB format."
def getArgumentParser(parser = ap.ArgumentParser(description = DESCRIPTION)):
	parser.add_argument('-s', '--subject',
		nargs = 1,
		metavar = ('str'),
		help='Path to the recon-all results. Default: %(default)s)',
		required = True)
	parser.add_argument("-sjd", "--settingsjsondefaults",
		nargs = 1,
		help = "JSON settings file (default: ./processing_settings.json)",
		metavar = ('str'))
	parser.add_argument("-n_jobs",
		nargs = 1,
		default = ['12'],
		help = "Number of threads",
		metavar = ('int'))
	return parser

def run(opts):

	subject = opts.subject[0]
	n_jobs = int(opts.n_jobs[0])

	if opts.settingsjsondefaults:
		json_settings_file = opts.settingsjsondefaults[0]
	else:
		json_settings_file = ("%s/processing_settings.json" % os.path.dirname(os.path.realpath(__file__)))

	with open(json_settings_file) as json_file:
		processing_settings = json.load(json_file)

	dwi_output_directory = processing_settings['dwi_output_directory']
	recon_all_dir = processing_settings['recon_all_dir']
	tvb_output_dir = processing_settings['tvb_output']
	connectome_file_names = processing_settings['connectome_file_names']

	parcellation = processing_settings["parcellation"]
	if parcellation == "hcpmmp1":
		borked_names = processing_settings["mrtrix_conversion"]["borked_names"]
		fixed_names = processing_settings["mrtrix_conversion"]["fixed_names"]
	label_info_file = processing_settings["mrtrix_conversion"]["label_info"]

	parc_image = ("%s/%s/%s"  % (dwi_output_directory,
					subject,
					processing_settings["default_file_names"]["parc_image"]))
	tracts_path = ("%s/%s/%s"  % (dwi_output_directory,
					subject,
					processing_settings["default_file_names"]["tracts"]))
	weights_path = ("%s/%s/%s"  % (dwi_output_directory,
					subject,
					processing_settings["default_file_names"]["weights"]))

	tvb_output_path = "%s/sub-%s" % (tvb_output_dir, subject)
	if not os.path.exists(tvb_output_path):
		os.makedirs(tvb_output_path)

	tvb_output = {}


	tvb_output["EEG_Projection_mat"] = "%s/sub-%s_EEGProjection.mat" % (tvb_output_path, subject)
	tvb_output["EEG_Projection_npy"] = "%s/sub-%s_EEGProjection.npy" % (tvb_output_path, subject)
	tvb_output["EEG_Locations"] = "%s/sub-%s_EEG_Locations.txt" % (tvb_output_path, subject)


	tvb_output["region_mapping"] = "%s/sub-%s_region_mapping.txt" % (tvb_output_path, subject)
	tvb_output["region_mapping_hires"] = "%s/sub-%s_region_mapping_hires.txt" % (tvb_output_path, subject)


	# Fix labels and get centroids and area from parcellation image (voxel)
	parc_img = nib.load(parc_image)
	parc_data = parc_img.get_data()
	lables, labels_sizes, label_mean_coordinates = parc_get_mean_coord(parc_data, parc_img.affine, closest_value_voxel = True)

	label_info = np.genfromtxt(label_info_file, dtype=str)
	label_names = []
	for label in lables:
		label_names.append(label_info[:,1][label_info[:,0] == str(int(label))])
	label_names = np.array(label_names).flatten()

	if parcellation == "hcpmmp1":
		for i, corr_label in enumerate(fixed_names):
			if borked_names[i] in label_names:
				print(borked_names[i] + "->" + corr_label)
				label_names[label_names == borked_names[i]] = corr_label

		region_names = label_names
		cortical = np.zeros((len(region_names)))
		cortical[:358] = 1
		hemisphere = [ 1 if name[0] == "R" else 0 for name in region_names]

	# process surfaces
	pial_surf = ['%s/%s/surf/lh.pial' % (recon_all_dir, subject),
					'%s/%s/surf/rh.pial' % (recon_all_dir, subject)]
	if parcellation == "hcpmmp1":
		annots = ['%s/%s/label/lh.%s_HCP-MMP1.annot' % (recon_all_dir, subject, subject),
					'%s/%s/label/rh.%s_HCP-MMP1.annot' % (recon_all_dir, subject, subject)]
	cortex_label = ['%s/%s/label/lh.cortex.label' % (recon_all_dir, subject),
					'%s/%s/label/rh.cortex.label' % (recon_all_dir, subject)]

	# make a surface output directory
	output_surf_directory = "%s/surf" % tvb_output_path
	if not os.path.exists(output_surf_directory):
		os.makedirs(output_surf_directory)

	v, f, s_info, region_map, surface_labels = subset_surface_by_cortex_label(surfs = pial_surf, 
											annots = annots, 
											cortex_labels = cortex_label,
											subset_labels = label_names,
											output_surf_directory = output_surf_directory)

	s_info_fs = s_info

	#%%
	# =============================================================================
	# compute source space 
	# =============================================================================
	# decimate surface

	pial_dec = mne.decimate_surface(v, f, n_triangles = 30000)
	nib.freesurfer.io.write_geometry('%s/cortex_dec.srf' % output_surf_directory, pial_dec[0], pial_dec[1], volume_info = s_info_fs)

	# complete decimated surface (add normals + other parameters)
	pial_dict = {'rr':pial_dec[0]/1000, 'tris':pial_dec[1]}
	pial_complete = mne.surface.complete_surface_info(pial_dict)

	# construct source space dictionary by hand
	# use all point of the decimated surface as souce space
	src =   {'rr':       pial_complete['rr'],
		     'tris':     pial_complete['tris'],
		     'ntri':     pial_complete['ntri'],
		     'use_tris': pial_complete['tris'],
		     'np':       pial_complete['np'],
		     'nn':       pial_complete['nn'],
		     'inuse':    np.ones(pial_complete['np']),
		     'nuse_tri': pial_complete['ntri'],
		     'nuse':     pial_complete['np'],
		     'vertno':   np.arange(0,pial_complete['np']),
		     'subject_his_id': subject,
		     'dist': None,
		     'dist_limit': None,
		     'nearest': None,
		     'type': 'surf',
		     'nearest_dist': None,
		     'pinfo': None,
		     'patch_inds': None,
		     'id': 101, # (FIFFV_MNE_SURF_LEFT_HEMI), # shouldn't matter, since we combined both hemispheres into one object
		     'coord_frame': 5} # (FIFFV_COORD_MRI)}
	src = mne.SourceSpaces([src])

	#%%
	# =============================================================================
	# compute BEM model + EEG Locations 
	# =============================================================================

	#  following line seems to work only when calling this script via command line, when executed within spyder it gives "Command not found: mri_watershed!"
	mne.bem.make_watershed_bem(subject = subject, subjects_dir = recon_all_dir, overwrite=True)
	# This will always fail

	# smooth and shrink the the inner skull surface a little bit. This prevents overlapping surface errors
	v_is, f_is, s_info = nib.freesurfer.read_geometry("%s/%s/bem/watershed/%s_inner_skull_surface" % (recon_all_dir, subject, subject), read_metadata=True)
	adj = create_adjac_vertex(v_is,f_is)
	v_smooth, f = vectorized_surface_smooth(v_is, f_is, adj, number_of_iter = 10, scalar = None, lambda_w = 0.5, mode = 'laplacian', weighted = True)

	nib.freesurfer.io.write_geometry('%s/%s/bem/inner_skull.surf' % (recon_all_dir, subject), v_smooth, f_is, volume_info = s_info)

	os.system("cp --remove-destination %s/%s/bem/watershed/%s_outer_skull_surface %s/%s/bem/outer_skull.surf" % (recon_all_dir, subject, subject, recon_all_dir, subject))
	os.system("cp --remove-destination %s/%s/bem/watershed/%s_outer_skin_surface %s/%s/bem/outer_skin.surf" % (recon_all_dir, subject, subject, recon_all_dir, subject))
	os.system("cp --remove-destination %s/%s/bem/watershed/%s_brain_surface %s/%s/bem/brain.surf" % (recon_all_dir, subject, subject, recon_all_dir, subject))

	conductivity = [0.3, 0.006, 0.3]  # for three layers
	try:
		model = mne.make_bem_model(subject = subject,
											ico = 4,
											conductivity = conductivity,
											subjects_dir = recon_all_dir,
											verbose = True)
	except:
		print("BEM model failed. Running laplacian smoothing on the inner surface (20 interations), and trying again.")
		v_is, f_is, s_info = nib.freesurfer.read_geometry("%s/%s/bem/watershed/%s_inner_skull_surface" % (recon_all_dir, subject, subject), read_metadata=True)
		adj = create_adjac_vertex(v_is,f_is)
		v_smooth, f = vectorized_surface_smooth(v_is, f_is, adj, number_of_iter = 20, scalar = None, lambda_w = 0.5, mode = 'laplacian', weighted = True)
		os.system("rm %s/%s/bem/inner_skull.surf" % (recon_all_dir, subject))
		nib.freesurfer.io.write_geometry('%s/%s/bem/inner_skull.surf' % (recon_all_dir, subject), v_smooth, f_is, volume_info = s_info)
		try:
			model = mne.make_bem_model(subject = subject,
												ico = 4,
												conductivity = conductivity,
												subjects_dir = recon_all_dir,
												verbose = True)
		except:
			print("BEM model failed. Running laplacian smoothing on the inner surface (40 interations), and trying again.")
			v_is, f_is, s_info = nib.freesurfer.read_geometry("%s/%s/bem/watershed/%s_inner_skull_surface" % (recon_all_dir, subject, subject), read_metadata=True)
			adj = create_adjac_vertex(v_is,f_is)
			v_smooth, f = vectorized_surface_smooth(v_is, f_is, adj, number_of_iter = 40, scalar = None, lambda_w = 0.5, mode = 'laplacian', weighted = True)
			os.system("rm %s/%s/bem/inner_skull.surf" % (recon_all_dir, subject))
			nib.freesurfer.io.write_geometry('%s/%s/bem/inner_skull.surf' % (recon_all_dir, subject), v_smooth, f_is, volume_info = s_info)
			model = mne.make_bem_model(subject = subject,
												ico = 4,
												conductivity = conductivity,
												subjects_dir = recon_all_dir,
												verbose = True)
	bem = mne.make_bem_solution(model)



	### GET AND ADJUST EEG LOCATIONS FROM DEFAULT CAP !!!!!! 
	# This may produce implausible results if not corrected. 
	# Default locations are used here to completely automize the pipeline and to not require manual input (e.g. setting the fiducials and fitting EEG locations.)
	# read default cap
#	mon = mne.channels.read_montage(kind="biosemi64", unit='auto', transform=True)
	mon = mne.channels.make_standard_montage(kind="biosemi64")
	#mon = mne.channels.read_montage(kind="easycap-M1", unit='auto', transform=False)

	# create info object
	ch_type = ["eeg" for i in range(len(mon.ch_names))]
#	ch_type[-3:] = ["misc", "misc", "misc"] # needed for caps which include lpa, rpa and nza "channels" at the end
	info = mne.create_info(ch_names = mon.ch_names, sfreq = 256, ch_types = ch_type, montage=mon)

	# load head surface
	try:
		surf = mne.get_head_surf(subject = subject,
										source = "head",
										subjects_dir = recon_all_dir)
	except:
		bem_dir = os.path.join(recon_all_dir, subject, 'bem')
		ws_dir = os.path.join(recon_all_dir, subject, 'bem', 'watershed')
		fname_head = os.path.join(bem_dir, subject + '-head.fif')
		if os.path.isfile(fname_head):
			os.remove(fname_head)
		surf = _surfaces_to_bem([os.path.join(ws_dir, subject + '_outer_skin_surface')],[FIFF.FIFFV_BEM_SURF_ID_HEAD], sigmas=[1])
		write_bem_surfaces(fname_head, surf)
		surf = mne.get_head_surf(subject = subject,
										source = "head",
										subjects_dir = recon_all_dir)


	# project eeg locations onto surface and save into info
	eeg_loc = np.array([info['chs'][i]['loc'][:3] for i in range(len(mon.ch_names))])
	eegp_loc, eegp_nn = _project_onto_surface(eeg_loc,
															surf,
															project_rrs = True,
															return_nn = True)[2:4]
	for i in range(len(mon.ch_names)):
		info['chs'][i]['loc'][:3] = eegp_loc[i,:]

	#%% 
	# =============================================================================
	# compute forward solution
	# =============================================================================

	fwd = mne.make_forward_solution(info,
												trans = None,
												src = src,
												bem = bem,
												meg = False,
												eeg = True,
												mindist = 0.0,
												n_jobs = n_jobs)

	fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
	leadfield = fwd_fixed['sol']['data']

	# write leadfield to file
	sio.savemat(tvb_output["EEG_Projection_mat"], mdict={'ProjectionMatrix':leadfield})
	np.save(tvb_output["EEG_Projection_npy"], leadfield)

	#%% 
	# =============================================================================
	# save files for TVB
	# =============================================================================
	# get region map for source space (ie. downsampled pial), via nearest neighbour interpolation

	n_vert = pial_complete['np']
	region_map_lores = np.zeros((n_vert))
	vert_hires = v/1000
	vert_lores = pial_complete['rr']

	# serach nearest neighbour
	idx = spatial.KDTree(vert_hires).query(vert_lores)[1]
	region_map_lores = region_map[idx]

	np.savetxt(tvb_output["region_mapping"], region_map_lores, fmt="%i")
	np.savetxt(tvb_output["region_mapping_hires"], region_map, fmt="%i")


	# write low res labels

	write_out_labels(surface_labels = surface_labels,
					vert_region_map = region_map_lores,
					surface_file = '%s/cortex_dec.srf' % output_surf_directory,
					output_directory = '%s/decimated_labels_hcpmmp1' % output_surf_directory,
					parcellation = "hcpmmp1",
					subject = subject)

	# make annot
	os.system("cp %s/cortex_dec.srf %s/%s/surf/lh.cortex_dec.srf" % (output_surf_directory,recon_all_dir, subject))
	label_files_long = " --l ".join(glob.glob("%s/decimated_labels_hcpmmp1/*label" % output_surf_directory))
	os.system("mris_label2annot --s %s --ctab %s --l %s --annot-path %s/hcpmmp1_dec.annot --surf cortex_dec.srf --h lh --sd %s" % (subject, label_info_file, label_files_long, output_surf_directory, recon_all_dir))
	os.system("rm %s/%s/surf/lh.cortex_dec.srf" % (recon_all_dir, subject))


	# write BEM surfaces in RAS
	write_surf_tvb(in_surf_file = '%s/%s/bem/brain.surf' % (recon_all_dir, subject),
						out_surf_file = "%s/bem_brain.srf" % output_surf_directory,
						s_info = s_info_fs)

	write_surf_tvb(in_surf_file = '%s/%s/bem/inner_skull.surf' % (recon_all_dir, subject),
						out_surf_file = "%s/bem_inner_skull.srf" % output_surf_directory,
						s_info = s_info_fs)

	write_surf_tvb(in_surf_file = '%s/%s/bem/outer_skull.surf' % (recon_all_dir, subject),
						out_surf_file = "%s/bem_outer_skull.srf" % output_surf_directory,
						s_info = s_info_fs)

	write_surf_tvb(in_surf_file = '%s/%s/bem/outer_skin.surf' % (recon_all_dir, subject),
						out_surf_file = "%s/bem_outer_skin.srf" % output_surf_directory,
						s_info = s_info_fs)




	# [leave this alone for now.]
	# write cortical surface (i.e. source space) to file
	cort_surf_path = tvb_output_path+"/sub-"+subject+"_Cortex/"
	if not os.path.exists(cort_surf_path):
		os.makedirs(cort_surf_path)

	# surface vertices are in ras-tkr coordinates used by freesurfer
	# for them to allign with parc_image, use affine transform to bring them into ras-scanner
	p = Popen(('mri_info --tkr2scanner '+recon_all_dir+"/"+subject+"/mri/aparc+aseg.mgz").split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
	output, err = p.communicate(b"input data that is passed to subprocess' stdin")
	affine_xfm = np.array([ i.split() for i in str(output, "utf-8").splitlines()], dtype="float")

	pial_vert_converted = affine_xfm.dot(np.concatenate((pial_complete['rr'] * 1000 ,np.ones((pial_complete['rr'].shape[0],1))), axis=1).T)[:3,:].T

	# save
	np.savetxt(cort_surf_path+"triangles.txt", pial_complete['tris'], fmt="%i")
	np.savetxt(cort_surf_path+"vertices.txt", pial_vert_converted, fmt="%1.6f")
	np.savetxt(cort_surf_path+"normals.txt", pial_complete['nn'], fmt="%1.6f")


	# zip files
	shutil.make_archive(cort_surf_path[:-1], 'zip', cort_surf_path)
	print("Cortical surface zipped !")
	shutil.rmtree(cort_surf_path)


	# write BEM surfaces too file
	names = ["inner_skull_surface", "outer_skull_surface", "outer_skin_surface"] # "brain_surface",
	for name in names :
		# make dir
		bem_path = tvb_output_path+"/sub-"+subject+"_"+name+"/"
		if not os.path.exists(bem_path):
			os.makedirs(bem_path)
		bem_surf = mne.read_surface(recon_all_dir+"/"+subject+"/bem/watershed/"+subject+"_"+name)
		bem_dict = {'rr':bem_surf[0], 'tris':bem_surf[1]}
		bem_complete = mne.surface.complete_surface_info(bem_dict)
		
		bem_vert_converted = affine_xfm.dot(np.concatenate((bem_complete['rr'] ,np.ones((bem_complete['rr'].shape[0],1))), axis=1).T)[:3,:].T

		# save files
		np.savetxt(bem_path+"triangles.txt", bem_complete['tris'], fmt="%i")
		np.savetxt(bem_path+"vertices.txt", bem_vert_converted, fmt="%1.6f")
		np.savetxt(bem_path+"normals.txt", bem_complete['nn'], fmt="%1.6f")
		
		# zip folder
		shutil.make_archive(bem_path[:-1], 'zip', bem_path)
		shutil.rmtree(bem_path)
	print("BEM surfaces saved  !")

	# save eeg_locations, are in ras-tkr coordinates used by freesurfer
	# for them to allign with parc_image, use affine transform to bring them into ras-scanner
	eegp_loc_converted = affine_xfm.dot(np.concatenate((eegp_loc * 1000 ,np.ones((eegp_loc.shape[0],1))), axis=1).T)[:3,:].T

	f = open(tvb_output["EEG_Locations"], "w")
	for i in range(len(eegp_loc_converted)):
		f.write(mon.ch_names[i]+" "+"%.6f" % eegp_loc_converted[i,0]+" "+"%.6f" %eegp_loc_converted[i,1]+" "+"%.6f" % eegp_loc_converted[i,2]+"\n")
	f.close()


	v_eeg, f_eeg = eeg_sensors_to_surface(eeg_coordinates = eegp_loc_converted, unit_size = 4, generate = False)
	nib.freesurfer.io.write_geometry('%s/eeg_biosemi64.srf' % (output_surf_directory), v_eeg, f_eeg)

	print("EEG locations saved  !")

	# create and save connectome.zip
	tvb_connectome_path = tvb_output_path+"/sub-"+subject+"_Connectome/"
	if not os.path.exists(tvb_connectome_path):
		os.makedirs(tvb_connectome_path)


	# 1 weights, set diagonal to zero and make it symmetric
	weights = np.genfromtxt(weights_path)
	weights[np.diag_indices_from(weights)] = 0
	i_lower = np.tril_indices_from(weights, -1)
	weights[i_lower] = weights.T[i_lower]
	n_regions = weights.shape[0]
	np.savetxt(tvb_connectome_path + connectome_file_names['weights'], weights, delimiter="\t")

	print("Weights saved  !")

	# 2 tracts, set diagonal to zero and make it symmetric
	tracts  = np.genfromtxt(tracts_path)
	tracts[np.diag_indices_from(tracts)] = 0
	i_lower = np.tril_indices_from(tracts, -1)
	tracts[i_lower] = tracts.T[i_lower]
	np.savetxt(tvb_connectome_path + connectome_file_names['tract_lengths'], tracts, delimiter="\t")
	print("Tracts saved !")

	#3. centers
	with open(tvb_connectome_path + connectome_file_names['centres'], "a") as roicentres:
		for i, roi in enumerate(label_names):
			roicentres.write("%s\t%1.6f\t%1.6f\t%1.6f\n" % (roi, label_mean_coordinates[i,0],label_mean_coordinates[i,1], label_mean_coordinates[i,2]))
		voxel_area = parc_img.header['pixdim'][1] * parc_img.header['pixdim'][1] * parc_img.header['pixdim'][1]
	print("Centers saved !")

	# 4 orientation
	# First get all Vertex-Normals corresponding to the Vertices of a Region 
	# Now compute mean Vector and Normalize the Vector
	# for subcortical regions set [0,0,1]
	orientation = np.zeros((n_regions,3))
	v_smooth, f, _ =  nib.freesurfer.read_geometry('%s/cortex_smooth.srf' % output_surf_directory, read_metadata=True)
	n = compute_normals(v_smooth, f)
	for i in range(n_regions):
		if cortical[i] == 1: # cortical regions
			nn  = n[region_map==int(i + 1),:]
			orientation[i,:] = nn.mean(axis=0)/np.linalg.norm(nn.mean(axis=0))
		elif not cortical[i] == 1:  # subcortical regions
			# select normal vertices of a region, average and normalize them
			orientation[i,:] = np.array([0,0,1])
	np.savetxt(tvb_connectome_path + connectome_file_names['average_orientations'], orientation, fmt="%f")
	print("Orientations saved !")

	# 5 area
	area = get_surface_area(surface_file = '%s/cortex_smooth.srf' % output_surf_directory,
									region_map = region_map,
									verbose = True)
	np.savetxt(tvb_connectome_path + connectome_file_names['areas'], area, fmt='%1.6f')
	print("Area saved !")

	# 6 cortical
	# connectivity cortical/non-cortical region flags; text file containing one boolean value on each line 
	# (as 0 or 1 value) being 1 when corresponding region is cortical.
	# due to error in configuring projection matrix in EEG, see monitors.py, class Projection, def config_for_sim
	# this would need to get fixed, otherwise I don't know how to define the cortical variable or the lead field matrix
	# therefor for now declare all regions as cortical
	np.savetxt(tvb_connectome_path + connectome_file_names['cortical'], cortical, fmt="%i")
	print("Cortical saved !")


	# 7 hemisphere
	# text file containing one boolean value on each line 
	# (as 0 or 1 value) being 1 when corresponding region is in the right hemisphere and 0 when in left hemisphere.
	np.savetxt(tvb_connectome_path +  connectome_file_names['hemispheres'], hemisphere, fmt="%i")
	print("Hemisphere saved !")

	# zip all files
	shutil.make_archive(tvb_connectome_path[:-1], 'zip', tvb_connectome_path)
	print("Connectome zipped !")
	shutil.rmtree(tvb_connectome_path)

	print("TVBconverter has finished !")

if __name__ == "__main__":
	parser = getArgumentParser()
	opts = parser.parse_args()
	run(opts)
