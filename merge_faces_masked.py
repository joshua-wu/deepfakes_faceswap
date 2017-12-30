import argparse
import cv2
import json
import numpy
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoder_A, decoder_B

encoder  .load_weights( "models/encoder.h5"   )
decoder_A.load_weights( "models/decoder_A.h5" )
decoder_B.load_weights( "models/decoder_B.h5" )


def convert_one_image( autoencoder, image, mat,facepoints,erosion_kernel,blurSize,seamlessClone,maskType ):
    size = 64
    image_size = image.shape[1], image.shape[0]

    face = cv2.warpAffine( image, mat * size, (size,size) )
    face = numpy.expand_dims( face, 0 )
    new_face = autoencoder.predict( face / 255.0 )[0]

    new_face = numpy.clip( new_face * 255, 0, 255 ).astype( image.dtype )

    face_mask = numpy.zeros(image.shape,dtype=float)
    if 'Rect' in maskType:
      face_src = numpy.ones(new_face.shape,dtype=float) 
      cv2.warpAffine( face_src, mat * size, image_size, face_mask, cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT )

    hull_mask = numpy.zeros(image.shape,dtype=float)
    if 'Hull' in maskType:
      hull = cv2.convexHull( numpy.array( facepoints ).reshape((-1,2)).astype(int) ).flatten().reshape( (-1,2) )
      cv2.fillConvexPoly( hull_mask,hull,(1,1,1) )

    if maskType == 'FaceHull':
      image_mask = hull_mask
    elif maskType == 'Rect':
      image_mask = face_mask
    else:
      image_mask = ((face_mask*hull_mask))


    if erosion_kernel is not None:
      image_mask = cv2.erode(image_mask,erosion_kernel,iterations = 1)

    if blurSize!=0:
      image_mask = cv2.blur(image_mask,(blurSize,blurSize))

    base_image = numpy.copy( image )
    new_image = numpy.copy( image )

    cv2.warpAffine( new_face, mat * size, image_size, new_image, cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT )

    outImage = None
    if seamlessClone:
      masky,maskx = cv2.transform( numpy.array([ size/2,size/2 ]).reshape(1,1,2) ,cv2.invertAffineTransform(mat*size) ).reshape(2).astype(int)
      outimage = cv2.seamlessClone(new_image.astype(numpy.uint8),base_image.astype(numpy.uint8),(image_mask*255).astype(numpy.uint8),(masky,maskx) , cv2.NORMAL_CLONE )
    else:
      foreground = cv2.multiply(image_mask, new_image.astype(float))
      background = cv2.multiply(1.0 - image_mask, base_image.astype(float))
      outimage = cv2.add(foreground, background)

    return outimage

def main( args ):




    input_dir = Path( args.input_dir )
    assert input_dir.is_dir()

    alignments = input_dir / args.alignments
    with alignments.open() as f:
        alignments = json.load(f)

    output_dir = input_dir / args.output_dir
    output_dir.mkdir( parents=True, exist_ok=True )

    if args.direction == 'AtoB': autoencoder = autoencoder_B
    if args.direction == 'BtoA': autoencoder = autoencoder_A
    
    if args.erosionKernelSize>0:
      erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(args.erosionKernelSize,args.erosionKernelSize))
    else:
      erosion_kernel = None

    for e in alignments:    
      if len(e)<4:
        raise LookupError('This script expects new format json files with face points included.')


    for image_file, face_file, mat,facepoints in tqdm( alignments ):
        image = cv2.imread( str( input_dir / image_file ) )
        face  = cv2.imread( str( input_dir / face_file  ) )


        mat = numpy.array(mat).reshape(2,3)

        if image is None: continue
        if face  is None: continue


        new_image = convert_one_image( autoencoder, image, mat, facepoints, erosion_kernel, args.blurSize, args.seamlessClone, args.maskType)

        output_file = output_dir / Path(image_file).name
        cv2.imwrite( str(output_file), new_image )

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "input_dir", type=str )
    parser.add_argument( "alignments", type=str, nargs='?', default='alignments.json' )
    parser.add_argument( "output_dir", type=str, nargs='?', default='merged' )

    parser.add_argument("--seamlessClone", type=str2bool, nargs='?', const=False, default='False', help="Attempt to use opencv seamlessClone.")

    parser.add_argument('--maskType', type=str, default='FaceHullAndRect' ,choices=['FaceHullAndRect','FaceHull','Rect'], help="The type of masking to use around the face.")

    parser.add_argument( "--blurSize",          type=int, default='2' )
    parser.add_argument( "--erosionKernelSize", type=int, default='0' )
    parser.add_argument( "--direction",         type=str, default="AtoB", choices=["AtoB", "BtoA"])
    main( parser.parse_args() )

