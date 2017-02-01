# inkscapeOptimizePath
An Inkscape extension that tries to make the longest paths possible

It converts the path to a graph and the use a depth first search algorithm to traverse the edges and build long connected paths.
Was written for the [Axidraw](http://axidraw.com) pen plotter to reduce writing time of graph-like drawings like Voronoi diagrams and meshes.

---------

**Usage**
- Make sure to ungroup everything
- Make sure that the paths use only absolute coordinates (see trick below)
- Make sure to not have transforms on the paths. You can use the [Apply Transform](https://inkscape.org/en/~Klowner/★apply-transforms) extension to remove them
- Select all the paths you want to optimize (currently only works with lines)
- Open the extension ('Extensions > Modify Path > Optimize Path')
- Set the merge tolerance (0.1 should work in most cases)
- Press Apply

**Trick to remove relative coordinates**
1. Change the preferences for 'SVG Output > Path Data' to always use absolute coordinates. This will only affect newly created paths, or existing objects for which a rewrite of the path data is triggered.
2. For existing paths, use 'Edit > Select All in All Layers', and nudge the selection with the arrow keys (e.g. one step up and one back down again). This will trigger a rewrite of the path data in 'd' which will follow the changed preferences for optimized path data.


**Dependencies**
[NetworkX](http://networkx.github.io) a graph library which depends on [decorator](https://github.com/micheles/decorator)
