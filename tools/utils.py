import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import NearestNDInterpolator


def show_point_cloud(pc):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:,0], pc[:,1], pc[:,2], c='r', marker='o')
    plt.show()


def name_without_type(filename):
    spl = os.path.basename(filename).split('.')
    return spl[0]+spl[1]


def save_obj_file(pts, filepath):
    f = open(filepath, 'w')
    for v in xrange(pts.shape[0]):
        line = "v {} {} {}\n".format(pts[v, 0], pts[v, 1], pts[v, 2])
        f.write(line)
        if pts[v, :].shape[0] > 3:
            length = np.sqrt(np.sum(np.power(pts[v, 3:6], 2)))
            pts[v, 3:6] /= length
            line = "vn {} {} {}\n".format(pts[v, 3], pts[v, 4], pts[v, 5])
            f.write(line)
    f.close()


def save_objs(pts, path):
    filename = "pc_{}.obj"
    npts = pts.shape[1]
    ndims = pts.shape[3]

    for i in xrange(pts.shape[0]):
        save_obj_file(pts[i, :, :, :].reshape(npts, ndims),
                os.path.join(path, filename.format(str(i).zfill(4))))


def save_points(pts, path):
    filename = "pc_{}.npy"
    npts = pts.shape[1]
    ndims = pts.shape[3]

    for i in xrange(pts.shape[0]):
        np.save(path+"/"+filename.format(i),
                pts[i, :, :, :].reshape(npts, ndims))


def point_from_vline(vline):
    sp = vline.split(' ')[1:4]
    pt = []
    for v in sp:
        if np.isnan(float(v)):
            raise ValueError
        pt.append(float(v))
    return np.array(pt)


def read_obj(opath):
    points = []
    with open(opath) as f:
        for line in f:
            if line.startswith('vn'):
                points[-1] = np.append(points[-1], point_from_vline(line))
            else:
                points.append(point_from_vline(line))
    return np.array(points)


def load_objs(paths):
    allpoints = []
    for p in paths:
        allpoints.append(read_obj(p).flatten())
    return np.array(allpoints)


def load_bin_volume(f):
    s = np.array(np.fromfile(f, dtype=np.int8)).astype(float)
    dim = int(np.round(np.power(s.shape[0], 1/3.0)))
    s = (-s).reshape(dim, dim, dim)
    return s


def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben


def load_npy(filepaths, flatten=False):
    npys = []
    for f in filepaths:
        if flatten:
            npys.append(np.load(f).flatten())
        else:
            npys.append(np.load(f))
    return np.array(npys)


def load_flatten_imgbatch(img_paths):
    images = []
    for path in img_paths:
        images.append(mpimg.imread(path).flatten())
    return np.array(images)


def load_image(path):
    img = np.array(mpimg.imread(path))
    return img[:, :, 0:3]


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def save_image(image, image_path):
    return scipy.misc.imsave(image_path, image[0, :, :, 0])


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def imsave(images, size, path):
    return mpimg.imsave(path, merge(images, size))


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])


def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)


def inverse_transform(images):
    return np.clip(images, 0, 1)


def voxelize(samples, size=np.array([32., 32., 32.]), dims=np.array([1.0, 1.0, 1.0])):

    #newsamples = samples + np.random.multivariate_normal(
    #        np.zeros_like(samples.flatten()), 
    #        np.eye(samples.flatten().shape[0])*0.1).reshape(samples.shape)

    xs = np.linspace(-dims[0]/2., dims[0]/2., size[0]+1)
    ys = np.linspace(-dims[1]/2., dims[1]/2., size[1]+1)
    zs = np.linspace(-dims[2]/2., dims[2]/2., size[2]+1)

    voxels, _ = np.histogramdd(samples, bins=(xs, ys, zs))
    voxels = np.clip(voxels, 0, 1)

    return voxels


def center_samples(s):
    return s-np.mean(s, 0)


def volume_to_cubes(volume, threshold=0, dim=[2., 2., 2.]):
    o = np.array([-dim[0]/2., -dim[1]/2., -dim[2]/2.])
    step = np.array([dim[0]/volume.shape[0], dim[1]/volume.shape[1], dim[2]/volume.shape[2]])
    points = []
    faces = []
    for x in range(1, volume.shape[0]-1):
        for y in range(1, volume.shape[1]-1):
            for z in range(1, volume.shape[2]-1):
                pos = o + np.array([x, y, z]) * step
                if volume[x, y, z] > threshold:
                    vidx = len(points)+1
                    POS = pos + step*0.95
                    xx = pos[0]
                    yy = pos[1]
                    zz = pos[2]
                    XX = POS[0]
                    YY = POS[1]
                    ZZ = POS[2]
                    points.append(np.array([xx, yy, zz]))
                    points.append(np.array([xx, YY, zz]))
                    points.append(np.array([XX, YY, zz]))
                    points.append(np.array([XX, yy, zz]))
                    points.append(np.array([xx, yy, ZZ]))
                    points.append(np.array([xx, YY, ZZ]))
                    points.append(np.array([XX, YY, ZZ]))
                    points.append(np.array([XX, yy, ZZ]))
                    faces.append(np.array([vidx, vidx+1, vidx+2, vidx+3]))
                    faces.append(np.array([vidx, vidx+4, vidx+5, vidx+1]))
                    faces.append(np.array([vidx, vidx+3, vidx+7, vidx+4]))
                    faces.append(np.array([vidx+6, vidx+2, vidx+1, vidx+5]))
                    faces.append(np.array([vidx+6, vidx+5, vidx+4, vidx+7]))
                    faces.append(np.array([vidx+6, vidx+7, vidx+3, vidx+2]))
    return points, faces


def write_volume_obj(path, volume):
    pts, faces = volume_to_cubes(volume)
    write_cubes_obj(path, pts, faces)


def write_cubes_obj(path, points, faces):
    f = open(path, 'w')
    for p in points:
      f.write("v {} {} {}\n".format(p[0], p[1], p[2]))
    for q in faces:
      f.write("f {} {} {} {}\n".format(q[0], q[1], q[2], q[3]))


def points_to_cubes(point_cloud):
    size = 0.03
    points = []
    faces = []

    for i in xrange(point_cloud.shape[0]):
        vidx = len(points) + 1
        center = point_cloud[i, :]
        xx = center[0] - size
        yy = center[1] - size
        zz = center[2] - size
        XX = center[0] + size
        YY = center[1] + size
        ZZ = center[2] + size
        points.append(np.array([xx, yy, zz]))
        points.append(np.array([xx, YY, zz]))
        points.append(np.array([XX, YY, zz]))
        points.append(np.array([XX, yy, zz]))
        points.append(np.array([xx, yy, ZZ]))
        points.append(np.array([xx, YY, ZZ]))
        points.append(np.array([XX, YY, ZZ]))
        points.append(np.array([XX, yy, ZZ]))
        faces.append(np.array([vidx, vidx+1, vidx+2, vidx+3]))
        faces.append(np.array([vidx, vidx+4, vidx+5, vidx+1]))
        faces.append(np.array([vidx, vidx+3, vidx+7, vidx+4]))
        faces.append(np.array([vidx+6, vidx+2, vidx+1, vidx+5]))
        faces.append(np.array([vidx+6, vidx+5, vidx+4, vidx+7]))
        faces.append(np.array([vidx+6, vidx+7, vidx+3, vidx+2]))

    return points, faces


def scale_volume(volume, s):
   scaled_dims = np.array(volume.shape) * s 

   start = -0.5 + ((1.0/10)/2.0)
   end = 0.5 - ((1.0/10)/2.0)

   org_x = np.linspace(start, end, volume.shape[0])
   org_y = np.linspace(start, end, volume.shape[1])
   org_z = np.linspace(start, end, volume.shape[2])

   xs, ys, zs = np.meshgrid(org_x, org_y, org_z, indexing='ij')
   xyz = np.vstack((xs.flatten(), ys.flatten(), zs.flatten())).T
   interpolator = NearestNDInterpolator(xyz, volume.flatten())

   scaled_x = np.linspace(start, end, volume.shape[0]*s)
   scaled_y = np.linspace(start, end, volume.shape[1]*s)
   scaled_z = np.linspace(start, end, volume.shape[2]*s)
   scaled_xs, scaled_ys, scaled_zs = np.meshgrid(
           scaled_x, scaled_y, scaled_z, indexing='ij')
   scaled_xyz = np.vstack((scaled_xs.flatten(), 
       scaled_ys.flatten(), scaled_zs.flatten())).T

   return interpolator(scaled_xyz).reshape(scaled_dims.astype('int'))
