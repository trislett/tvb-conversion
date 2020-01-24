#!/usr/bin/env python

import os
import numpy as np
import nibabel as nib
from scipy import spatial


def create_adjac_vertex(vertices,faces): # basic version
	adjacency = [set([]) for i in range(vertices.shape[0])]
	for i in range(faces.shape[0]):
		adjacency[faces[i, 0]].add(faces[i, 1])
		adjacency[faces[i, 0]].add(faces[i, 2])
		adjacency[faces[i, 1]].add(faces[i, 0])
		adjacency[faces[i, 1]].add(faces[i, 2])
		adjacency[faces[i, 2]].add(faces[i, 0])
		adjacency[faces[i, 2]].add(faces[i, 1])
	return adjacency


def vectorized_surface_smooth(v, f, adjacency, number_of_iter = 5, scalar = None, lambda_w = 0.5, mode = 'laplacian', weighted = True):
	"""
	Applies Laplacian (Gaussian) or Taubin (low-pass) smoothing with option to smooth single volume
	
	Citations
	----------
	
	Herrmann, Leonard R. (1976), "Laplacian-isoparametric grid generation scheme", Journal of the Engineering Mechanics Division, 102 (5): 749-756.
	Taubin, Gabriel. "A signal processing approach to fair surface design." Proceedings of the 22nd annual conference on Computer graphics and interactive techniques. ACM, 1995.
	
	
	Parameters
	----------
	v : array
		vertex array
	f : array
		face array
	adjacency : array
		adjacency array

	
	Flags
	----------
	number_of_iter : int
		number of smoothing iterations
	scalar : array
		apply the same smoothing to a image scalar
	lambda_w : float
		lamda weighting of degree of movement for each iteration
		The weighting should never be above 1.0
	mode : string
		The type of smoothing can either be laplacian (which cause surface shrinkage) or taubin (no shrinkage)
		
	Returns
	-------
	v_smooth : array
		smoothed vertices array
	f : array
		f = face array (unchanged)
	
	Optional returns
	-------
	values : array
		smoothed scalar array
	
	"""
	k = 0.1
	mu_w = -lambda_w/(1-k*lambda_w)

	lengths = np.array([len(a) for a in adjacency])
	maxlen = max(lengths)
	padded = [list(a) + [-1] * (maxlen - len(a)) for a in adjacency]
	adj = np.array(padded)
	w = np.ones(adj.shape, dtype=float)
	w[adj<0] = 0.
	val = (adj>=0).sum(-1).reshape(-1, 1)
	w /= val
	w = w.reshape(adj.shape[0], adj.shape[1],1)

	vorig = np.zeros_like(v)
	vorig[:] = v
	if scalar is not None:
		scalar[np.isnan(scalar)] = 0
		sorig = np.zeros_like(scalar)
		sorig[:] = scalar

	for iter_num in range(number_of_iter):
		if weighted:
			vadj = v[adj]
			vadj = np.swapaxes(v[adj],1,2)
			weights = np.zeros((v.shape[0], maxlen))
			for col in range(maxlen):
				weights[:,col] = np.sqrt(np.linalg.norm(vadj[:,:,col] - v, axis=1))
			weights[adj==-1] = 0
			vectors = np.einsum('abc,adc->acd', weights[:,None], vadj)

			if scalar is not None:
				scalar[np.isnan(scalar)] = 0

				sadj = scalar[adj]
				sadj[adj==-1] = 0
				if lambda_w < 1:
					scalar = (scalar*(1-lambda_w)) + lambda_w*(np.sum(np.multiply(weights, sadj),axis=1) / np.sum(weights, axis = 1))
				else:
					scalar = np.sum(np.multiply(weights, sadj),axis=1) / np.sum(weights, axis = 1)
				scalar[np.isnan(scalar)] = sorig[np.isnan(scalar)] # hacky scalar nan fix
			if iter_num % 2 == 0:
				v += lambda_w*(np.divide(np.sum(vectors, axis = 1), np.sum(weights[:,None], axis = 2)) - v)
			elif mode == 'taubin':
				v += mu_w*(np.divide(np.sum(vectors, axis = 1), np.sum(weights[:,None], axis = 2)) - v)
			elif mode == 'laplacian':
				v += lambda_w*(np.divide(np.sum(vectors, axis = 1), np.sum(weights[:,None], axis = 2)) - v)
			else:
				print("Error: mode %s not understood" % mode)
				quit()
			v[np.isnan(v)] = vorig[np.isnan(v)] # hacky vertex nan fix
		else:
			if scalar is not None:
				sadj = scalar[adj]
				sadj[adj==-1] = 0

				if lambda_w < 1:
					scalar = (scalar*(1-lambda_w)) + (lambda_w*np.divide(np.sum(sadj, axis = 1),lengths))
				else:
					scalar = np.divide(np.sum(sadj, axis = 1),lengths)
			if iter_num % 2 == 0:
				v += np.array(lambda_w*np.swapaxes(w,0,1)*(np.swapaxes(v[adj], 0, 1)-v)).sum(0)
			elif mode == 'taubin':
				v += np.array(mu_w*np.swapaxes(w,0,1)*(np.swapaxes(v[adj], 0, 1)-v)).sum(0)
			elif mode == 'laplacian':
				v += np.array(lambda_w*np.swapaxes(w,0,1)*(np.swapaxes(v[adj], 0, 1)-v)).sum(0)
			else:
				print("Error: mode %s not understood" % mode)
				quit()

	if scalar is not None:
		return (v, f, scalar)
	else:
		return (v, f)

def parc_get_mean_coord(data, affine, closest_value_voxel = False):
	data = np.array(data)
	if data.ndim == 4: # double check
		print("4D volume detected. Only the first volume will be displayed.")
		data = data[:,:,:,0]
	size_x, size_y, size_z = data.shape

	lables = np.unique(data)[1:]
	print("%d labels were detected" % len(lables))
	labels_sizes = []
	label_mean_coordinates = []
	for roi_id in lables:
		x,y,z = np.where(data==roi_id)
		coord = np.column_stack((x,y))
		coord = np.column_stack((coord,z))
		coord_array = nib.affines.apply_affine(affine, coord)
		labels_sizes.append(len(coord_array))
		if closest_value_voxel:
			middle = np.mean(coord_array,0)
			tree = spatial.KDTree(coord_array)
			dist, voxel_index = tree.query(middle)
			print("[%d] closest voxel distance: %1.3fmm" % (int(roi_id), dist))
			label_mean_coordinates.append(coord_array[voxel_index])
		else:
			label_mean_coordinates.append(np.mean(coord_array,0))
	labels_sizes = np.array(labels_sizes)
	label_mean_coordinates = np.array(label_mean_coordinates)
	return(lables, labels_sizes, label_mean_coordinates)

# stolen from TFCE mediation
def compute_normals(v, f):
	v_ = v[f]
	fn = np.cross(v_[:, 1] - v_[:, 0], v_[:, 2] - v_[:,0])
	fs = np.sqrt(np.sum((v_[:, [1, 2, 0], :] - v_) ** 2, axis = 2))
	fs_ = np.sum(fs, axis = 1) / 2 # heron's formula
	fa = np.sqrt(fs_ * (fs_ - fs[:, 0]) * (fs_ - fs[:, 1]) * (fs_ - fs[:, 2]))[:, None]
	vn = np.zeros_like(v, dtype = np.float32)
	vn[f[:, 0]] += fn * fa # weight by area
	vn[f[:, 1]] += fn * fa
	vn[f[:, 2]] += fn * fa
	vlen = np.sqrt(np.sum(vn ** 2, axis = 1))[np.any(vn != 0, axis = 1), None]
	vn[np.any(vn != 0, axis = 1), :] /= vlen
	return vn

def convert_fs(fs_surface, read_metadata = False):
	if read_metadata:
		v, f, s_info = nib.freesurfer.read_geometry(fs_surface, read_metadata = read_metadata)
		n = compute_normals(v, f)
		return(v, f, n, s_info)
	else:
		v, f = nib.freesurfer.read_geometry(fs_surface, read_metadata = read_metadata)
		n = compute_normals(v, f)
		return(v, f, n)

def convert_fslabel(name_fslabel):
	obj = open(name_fslabel)
	reader = obj.readline().strip().split()
	reader = np.array(obj.readline().strip().split())
	if reader.ndim == 1:
		num_vertex = reader[0].astype(np.int)
	else:
		print('Error reading header')
	v_id = np.zeros((num_vertex)).astype(np.int)
	v_ras = np.zeros((num_vertex,3)).astype(np.float)
	v_value = np.zeros((num_vertex)).astype(np.float)
	for i in range(num_vertex):
		reader = obj.readline().strip().split()
		v_id[i] = np.array(reader[0]).astype(np.int)
		v_ras[i] = np.array(reader[1:4]).astype(np.float)
		v_value[i] = np.array(reader[4]).astype(np.float)
	return (v_id, v_ras, v_value)

def reduce_surface(v, f, v_subset):
	"""
	Subset a surface (v, f) by vertex indices (v_subset) and return reordered surface

	Parameters
	----------
	v : ndarray
		(Nvertices, 3)
	f : ndarray
		(Nfaces, 3)
	v_index : ndarray
		(Nvertices, 3)
	Returns
	---------
	v_out : ndarray
		(Nvertices, 3)
	f_out : ndarray
		(Nfaces, 3)
	
	"""
	v_index = np.zeros_like(v[:,0])
	v_index[v_subset] = 1
	missing = np.argwhere(v_index!=1)
	face_subset = np.ones(len(f))
	for i in missing:
		face_subset[f[:,0] == i] = -1
		face_subset[f[:,1] == i] = -1
		face_subset[f[:,2] == i] = -1
	f_new = f[face_subset==1]
	from_values = np.unique(f_new)
	to_values = np.arange(from_values.size)
	sort_idx = np.argsort(from_values)
	idx = np.searchsorted(from_values, 
								f_new,
								sorter = sort_idx)
	f_out = to_values[sort_idx][idx]
	v_out =  v[v_subset]
	return(v_out, f_out)

def subset_surface_by_cortex_label(surfs, annots, cortex_labels, subset_labels, output_surf_directory, n_smooth_iter = 10, return_smoothed = True):

	vl_id = convert_fslabel(cortex_labels[0])[0]
	l_labels, l_ctab, l_names  = nib.freesurfer.read_annot(annots[0])
	vl, fl, s_info = nib.freesurfer.read_geometry(surfs[0], read_metadata = True)

	surface_labels = []
	l_new_labels = -np.ones_like(l_labels)
	for i, l_name in enumerate(l_names):
		c_index = -1
		try:
			l_name = l_name.decode()
		except (UnicodeDecodeError, AttributeError):
			pass
		if l_name.endswith('_ROI'):
			l_name = l_name[:-4]
		try:
			c_index = np.argwhere(subset_labels == l_name)[0][0] + 1
		except:
			c_index = -1
		l_new_labels[l_labels == i + 1] = c_index
		print(l_name, c_index)
		if c_index!=-1:
			surface_labels.append(l_name)

	vl_new, fl_new = reduce_surface(v = vl,
											f = fl,
											v_subset = vl_id)

	vr_id = convert_fslabel(cortex_labels[1])[0]
	r_labels, r_ctab, r_names  = nib.freesurfer.read_annot(annots[1])
	vr, fr, nr = convert_fs(surfs[1]) 

	r_new_labels = -np.ones_like(r_labels)
	for i, r_name in enumerate(r_names):
		c_index = -1
		try:
			r_name = r_name.decode()
		except (UnicodeDecodeError, AttributeError):
			pass
		if r_name.endswith('_ROI'):
			r_name = r_name[:-4]
		try:
			c_index = np.argwhere(subset_labels == r_name)[0][0] + 1
		except:
			c_index = -1
		print(r_name, c_index)
		r_new_labels[r_labels == i + 1] = c_index
		if c_index!=-1:
			surface_labels.append(r_name)

	vr_new, fr_new = reduce_surface(v = vr,
											f = fr,
											v_subset = vr_id)

	v = np.row_stack((vl_new, vr_new))
	f = np.row_stack((fl_new, (fr_new + fl_new.max() + 1)))
	region_map = np.concatenate((l_new_labels[vl_id], r_new_labels[vr_id]))
	nib.freesurfer.io.write_geometry('%s/cortex.srf' % output_surf_directory, v, f, volume_info=s_info)

	if return_smoothed:
		adj = create_adjac_vertex(v, f)
		v_smooth, f = vectorized_surface_smooth(v, f, adj, number_of_iter = n_smooth_iter, scalar = None, lambda_w = 0.5, mode = 'laplacian', weighted = True)
		nib.freesurfer.io.write_geometry('%s/cortex_smooth.srf' % output_surf_directory, v_smooth, f, volume_info=s_info)
		return(v_smooth, f, s_info, region_map, surface_labels)
	else:
		return(v, f, s_info, region_map, surface_labels)

def convert_surface_to_ras(v, s_info):
	affine_xfm = np.eye(4)
	affine_xfm[:3,3] =  s_info['cras']
	v_converted = affine_xfm.dot(np.concatenate((v,np.ones((v.shape[0],1))), axis=1).T)[:3,:].T
	return(v_converted)

def get_surface_area(surface_file, region_map, verbose = True):
	v, f, s_info = nib.freesurfer.read_geometry(surface_file, read_metadata=True)
	v_ = v[f]
	fs = np.sqrt(np.sum((v_[:, [1, 2, 0], :] - v_) ** 2, axis = 2))
	fs_ = np.sum(fs, axis = 1) / 2 # heron's formula
	fa = np.sqrt(fs_ * (fs_ - fs[:, 0]) * (fs_ - fs[:, 1]) * (fs_ - fs[:, 2]))[:, None]

	region_index = np.unique(region_map)
	region_areas = []
	for roi in region_index:
		if roi != -1:
			temp = np.argwhere(region_map == roi)
			a = np.isin(f[:,0],temp.squeeze())
			b = np.isin(f[:,1],temp.squeeze())
			c = np.isin(f[:,2],temp.squeeze())
			index_temp = a*b*c
			area = np.sum(fa[index_temp])
			region_areas.append(area)
			if verbose:
				print("Region #: %d\t(Vertices, Faces): (%d,%d)\tArea(mm^2): %1.4f\t" %(roi, len(v[region_map == roi]), len(f[index_temp]), area))
	return(np.array(region_areas))

def write_out_labels(surface_labels, vert_region_map, surface_file, output_directory, parcellation = "hcpmmp1", subject = "??"):

	if not os.path.exists(output_directory):
		os.system("mkdir -p %s" % output_directory)
	v, f, s_info = nib.freesurfer.read_geometry(surface_file, read_metadata=True)
	v = convert_surface_to_ras(v, s_info)
	regions = np.unique(vert_region_map)
	if regions[0] == -1:
		regions = regions[1:]
	for i, roi in enumerate(regions):
		roi_name = surface_labels[roi-1]
		print(i, roi, roi_name)
		vert_roi = v[vert_region_map == roi]
		vert_index = np.argwhere(vert_region_map == roi).squeeze() 
		n_vert = len(vert_roi)
		f = open("%s/%s.label" % (output_directory, roi_name), "w")
		f.write("#!ascii label  , from %s vox2ras=TkReg\n" % subject)
		f.write("%d\n" % n_vert)
		if vert_roi.shape[0] == 1:
			f.write("%d %1.3f  %1.3f  %1.3f 0.0000000000\n" % (vert_index, vert_roi[0,0], vert_roi[0,1], vert_roi[0,2]))
		elif vert_roi.shape[0] > 1:
			for j in range(len(vert_roi)): # write only "eeg" electrodes (not "misc")
				f.write("%d %1.3f  %1.3f  %1.3f 0.0000000000\n" % (vert_index[j], vert_roi[j,0], vert_roi[j,1], vert_roi[j,2]))
		else:
			print("No indices found")
		f.close()


def spherical_polyhedron(n_vertices = 256):
	pts = np.arange(0, n_vertices, dtype=float) + 0.5
	phi = np.arccos(1 - 2 * pts/n_vertices)
	theta = np.pi * (1 + 5**0.5) * pts

	x =  np.cos(theta) * np.sin(phi) 
	y = np.sin(theta) * np.sin(phi)
	z = np.cos(phi)

	vertices = np.array([x, y, z]).T

	tri = spatial.Delaunay(vertices)
	faces = tri.simplices[:,1:]
	return(vertices, faces)


def eeg_sensors_to_surface(eeg_coordinates, unit_size = 5, generate = False):
	if generate: 
		vertex_array, faces_array = spherical_polyhedron()
	else:
		vertex_array, faces_array = nib.freesurfer.read_geometry('/mnt/mbServerProjects/SCRATCH/USERS/tris/CONNECTOME_INTEGRAMENT/TVB/tvb-conversion/sphere.srf')
	vertex_array = vertex_array * unit_size
	face_interator = 0
	out_v = []
	out_f = []
	for coord in eeg_coordinates:
		out_v.append(vertex_array + coord)
		out_f.append(faces_array + np.max(face_interator))
		face_interator = face_interator + np.max(faces_array) + 1
	out_v = np.concatenate(out_v)
	out_f = np.concatenate(out_f)
	return (out_v, out_f)

def write_surf_tvb(in_surf_file, out_surf_file, s_info):
	v_is, f_is = nib.freesurfer.read_geometry(in_surf_file)
	nib.freesurfer.io.write_geometry(out_surf_file, v_is, f_is, volume_info = s_info)




