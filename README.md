# Inkscape Optimize Path
An Inkscape extension that tries to make the longest continuous paths

It converts the paths to a graph, converts the graph to an Eulerian graph and finds an Eulerian cycle.
It was written for the [Axidraw](http://axidraw.com) pen plotter to reduce writing time of graph-like drawings like Voronoi diagrams and meshes.

[Example result](https://youtu.be/ZuaCT3Qi_-c)

---------

**Installation**
- Copy the files _optimizePaths.inx_ and _optimizePaths.py_ in the Inkscape extension folder (On MacOS: _/Applications/Inkscape.app/Contents/Resources/share/inkscape/extensions_)
- Download the latest [NetworkX](http://networkx.github.io) package and copy the _networkx_ folder in the Inkscape extension folder.
- Download the latest [Decorator](https://github.com/micheles/decorator/releases) package and copy the _decorator.py_ file in the Inkscape extension folder.

**Usage**
- Make sure to ungroup everything
- Make sure that the paths use only absolute coordinates (see trick below)
- Make sure to not have transforms on the paths. You can use the [Apply Transform](https://inkscape.org/en/~Klowner/★apply-transforms) extension to remove them
- Select all the paths you want to optimize (currently only works with poly-lines)
- Open the extension (Extensions > Generate from Path > Optimize Path)
- Set the merge tolerance (0.1 should work in most cases)
- Choose the Overwrite rule:
	- "Allow" means that the result will be a single path which might (probably) will have some overlapping edges.
	- "Allow none" means that the results will be multiple disconnected paths but there will be no overlapping edges.
	- "Allow some" is an in-between, overlapping edges are allowed, but only in short numbers. This is probably the best choice in most cases.
- Press Apply

**Trick to remove relative coordinates**
1. Change the preferences for 'SVG Output > Path Data' to always use absolute coordinates. This will only affect newly created paths, or existing objects for which a rewrite of the path data is triggered.

2. For existing paths, use 'Edit > Select All in All Layers', and nudge the selection with the arrow keys (e.g. one step up and one back down again). This will trigger a rewrite of the path data in 'd' which will follow the changed preferences for optimized path data.


**Dependencies**
[NetworkX](http://networkx.github.io), a graph library, which depends on [Decorator](https://github.com/micheles/decorator).
