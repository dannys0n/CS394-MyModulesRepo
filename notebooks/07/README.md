Naming convention for voronoi files:

k#-h#-a#<br>
k=seed count<br>
h=hotspots<br>
a=aspect ratio<br>
Each file’s voronoi are grouped by cases 0-7

Context:
========================================================================
A small model fine tuned with the goal of generating voronoi points that try to optimize for separating hotspots. <br>
We won’t be explicitly optimizing for load capacity, network conditions, and the cost of splitting hotspots in the context of each cell being a server in a server mesh.

Example Prompt:
========================================================================
```
You are placing shard seed centers in a normalized 2D world.

Coordinates are in [0,1] x [0,1]. The world aspect ratio is width/height.

Goal: Place K shard centers that balance load from hotspots.

Hotspots represent player density and include:

x, y position
weight (importance)
radius (influence area)
Guidelines:

Keep shard centers within [0,1] bounds
Prefer placing centers near clusters of hotspots
Avoid placing centers too close together
Try to distribute centers so hotspot load is balanced
Return ONLY valid JSON.

OUTPUT FORMAT: { "seeds": [ {"x": float, "y": float} ] }

INPUT_JSON: { "aspect_ratio": 0.5, "k": 4, "hotspots": [ {"x": 0.85, "y": 0.74, "weight": 0.32, "radius": 0.04}, {"x": 0.98, "y": 0.60, "weight": 0.31, "radius": 0.02}, {"x": 0.32, "y": 0.41, "weight": 0.18, "radius": 0.06} ] }
```
W&B loss curves:
========================================================================
Training itself looks healthy based on the graphs. Loss dropped quickly, learning rate decayed smoothly, grad norm stayed low after early steps. <br>
It appears with 1000 synthetic examples at an epoch of 3, it hovered at around 1.2.
<img width="770" height="313" alt="training" src="https://github.com/user-attachments/assets/0c7453c8-30b1-4ebf-994c-f22421720578" />
<img width="1090" height="336" alt="training2" src="https://github.com/user-attachments/assets/a3d8ad2c-7dd7-4644-80a8-3d3af83fefcd" />

What was expected:
========================================================================
The json structure would be strictly ahhered to. Output values would stay within the norrmalized constraints. The values would attempt to avoid clumping together if possible (not always).

What worked:
========================================================================
The requested json structure indeed was respected during my tests. Generally, these seeds looked pretty good when k value was less than h. This allowed seeds to separate, group, and split hotspots when they can. The seeds generated themselves produced some interesting results, as we will see below.

Starting with the most realistic setup for us. 4 servers and 6 hotspots.<br>
Looking at file k4-h6-a1, case 7 shows our best case scenario. nice separation of far away hotspots, grouping, and splitting very hot ones when necessary.<br>
<img width="497" height="501" alt="k4-h6-case7" src="https://github.com/user-attachments/assets/b012e607-91a3-488e-b199-1b77ad1c2459" />

Another ideal prompting scenario. 6 servers and 10 hotspots.<br>
Looking at file k6-h10-a1.5, case 2 also nicely grouped hotspots close together and separated when it can.<br>
<img width="608" height="449" alt="k4-h10-case2" src="https://github.com/user-attachments/assets/4048c3c4-a04a-4358-86c5-b103e456efe2" />

What didn’t:
========================================================================
It mostly came from a combination of k value being higher than h, and the python library for visualizing voronoi crashing from degenerate seeds. The failures would happen often when some seeds stack on top of each other from similar values, or are entirely colinear.

These issues can easily be resolved manually in a post-processing step. And while these issues were unexpected, they’re easy to detect for prune and jittering.

We can see these issues pop up in our test cases, where k was significantly higher than h.<br>
Looking at file k20-h10-a1, case 6 shows generally sound seeds, but then a lot of seeds went off into no mans land.<br>
<img width="446" height="442" alt="k20-h10-case6" src="https://github.com/user-attachments/assets/b7730135-13e3-4c93-9519-4c832e6a264f" />

Case 5 appears to separate quite nicely, but this was only after pruning seeds that were stacking on top of each other.<br>
<img width="448" height="449" alt="k20-h10-case5" src="https://github.com/user-attachments/assets/330b1cc2-97e6-409f-8ba9-7458a49a8752" />

What I would change:
========================================================================
Remove the aspect ratio parameter; it may have just introduced noise. We would assume a square aspect ratio for all completions. <br>
One possible extension would be to have a dedicated model also for 3d voronoi, I am curious of its reliability.

I would also consider prompting the synthetic generation step to avoid bad seeds, such as collinear and stacking.
