## Project 1: Coloring the Prokudin-Gorskii Photo Collection

<span style = "font-family=Papyrus; font-size:0.8em;"> 

Sergey Prokudin-Gorskii (aka. Prokudin-Gorsky) was a Russian chemist and photographer. 
From 1909 to 1915, he pioneered color photography with his three-plate 
photography, where he would take three exposures of the same image, using a red, blue and 
green filter. Then, when processing the image, he would combine the three "negatives" together 
to produce a complete color image.   
<br>
The task with this project is to replicate this exact process, only with code. To start, we are 
given the blue, green and red glass negatives stitched together in a single image file (in that 
order), which looks something like this:

<p>
</p>

<p align="center">
  <img src="cathedral.jpg" />

  <div align="center"> Figure 1: Stitched negatives of cathedral.jpg. From top to bottom: Blue, Green and 
  Red photos. </div> 
</p>

To start, we cut this image up into three sections, and call them `b`, `g` and `r`. These are 2D numpy arrays 
containing the intensity of each frequency of light. Next, we devise an algorithm that allows us to find the best 
possible alignment of the three negatives. To do this, we will use `cathedral.jpg` as our image to test. We select this 
image in particular for two reasons: first, there's regions with an abundance of features like the cathedral, and also 
regions which lack many features, like the clouds and also the field in the foreground. Second, it's also a relatively 
small image (300 x 300 pixels), so it's not computationally expensive. 

### The Metric

In order to find the best possible alignment, we need some heuristic (also called a metric) that allows us to quantify 
how well a given alignment between two plates `c1` and `c2` are. The easiest and simplest metric to use is the **Euclidean
Distance** metric, which essentially measures the difference in intensity of all pixels in `c1` against `c2`, summing 
over all pixels. In this case, the images will be most in line when this metric is minimized. 

In my experience working through this project, I've also found that this metric seemed to suffice and never really seemed 
like a bad heuristic to use to quantify alignment. I'm sure that there are more complex ones out there that probably do the 
job better, but I never really felt the need to deviate from the Euclidean distance. I will note that the project spec also 
mentions that we could have used a Normalized Cross-Correlation (NCC), but while doing this project I found that this was extremely obtuse 
to work with and didn't really work well for me.  

### Naive approach: exhaustive search

Perhaps the most obvious method we can find the best possible alignment is to do an exhaustive search of all possible 
alignments, then return the alignment `(i_opt, j_opt)` that minimizes the metric. While this does *guarantee* that we find the best possible 
alignment, this approach it falls flat since it's **extremely** expensive: consider the 300 x 300 pixel image: for every possible 
alignment there are 900 computations, and there are $300^2 = 90000$ possible alignments, so for this image alone there would be 
81 million operations needed!

Clearly, this is not the best approach. What does save us a little bit here is that the optimal alignment isn't very far off 
from just doing nothing, as we usually find that the optimal alignment only shifts a given plate by 10 or so. This means that we can 
reasonably search over a range of, say, `[-15, 15]` and be fairly certain that we have the optimal alignment, but of course 
this feels extremely hard-coded and unsatisfying. Therefore, we need a better approach.  
