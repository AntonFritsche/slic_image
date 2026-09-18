from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io, graph
import matplotlib.pyplot as plt
import argparse
import numpy as np


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))
# loop over the number of segments

def view_slic_segmentation(image: np.ndarray):
	for numSegments in (10, 100, 200, 300):
		# apply SLIC and extract (approximately) the supplied number
		# of segments
		segments = segment_image(numSegments)
		# show the output of SLIC
		fig = plt.figure("Superpixels -- %d segments" % numSegments)
		ax = fig.add_subplot(1, 1, 1)
		ax.imshow(mark_boundaries(image, segments, color=(1, 0, 0)))
		plt.axis("off")

	# show the plots
	plt.show()

def segment_image(num_segments: int, image: np.ndarray):
	return slic(image, n_segments=num_segments, sigma=5, compactness=10.0)

def build_rag_from_slic_segmentation(image: np.ndarray, segments: np.ndarray):
	rag_graph = graph.rag_mean_color(image, segments)

	fig, ax = plt.subplots(figsize=(6, 6))
	graph.show_rag(segments, rag_graph, image, ax=ax)
	plt.show()

if __name__ == "__main__":
	view_slic_segmentation(image)
	#build_rag_from_slic_segmentation(image, segment_image(100))