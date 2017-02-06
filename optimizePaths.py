#!/usr/bin/env python
'''
Copyright (C) 2017 Romain Testuz

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin St Fifth Floor, Boston, MA 02139
'''
import inkex, simplepath, simplestyle
import sys
import math
import random
import colorsys
import os
import numpy
#Trick to allow placing symbolic links in the inkscape extension folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import networkx as nx


"""
class Graph:
    def __init__(self):
        self.__adj = {}
        self.__data = {}

    def __str__(self):
        return str(self.__adj)

    def nodes(self):
        nodes = []
        for n in self.__adj:
            nodes.append(n)
        return nodes

    def edges(self):
        edges = []
        for n1 in self.__adj:
            for n2 in self.neighbours(n1):
                if((n2, n1) not in edges):
                    edges.append((n1, n2))
        return edges

    def node(self, n):
        if n in self.__adj:
            return self.__data[n]
        else:
            raise ValueError('Inexistant node')

    def neighbours(self, n):
        if n in self.__adj:
            return self.__adj[n]
        else:
            raise ValueError('Inexistant node')

    def outEdges(self, n):
        edges = []
        for n2 in self.neighbours(n):
            edges.append((n, n2))
        return edges

    def degree(self, n):
        if n in self.__adj:
            return len(self.__adj[n])
        else:
            raise ValueError('Inexistant node')

    def addNode(self, n, data):
        if n not in self.__adj:
            self.__adj[n] = []
            self.__data[n] = data
        else:
            raise ValueError('Node already exists')

    def removeNode(self, n):
        if n in self.__adj:
            #Remove all edges pointing to node
            for n2 in self.__adj:
                neighbours = self.__adj[n2]
                if n in neighbours:
                    neighbours.remove(n)
            del self.__adj[n]
            del self.__data[n]
        else:
            raise ValueError('Removing inexistant node')

    def addEdge(self, n1, n2):
        if(n1 in self.__adj and n2 in self.__adj):
            self.__adj[n1].append(n2)
            self.__adj[n2].append(n1)
        else:
            raise ValueError('Adding edge to inexistant node')

    def removeEdge(self, n1, n2):
        if(n1 in self.__adj and n2 in self.__adj and
        n2 in self.__adj[n1] and n1 in self.__adj[n2]):
            self.__adj[n1].remove(n2)
            self.__adj[n2].remove(n1)
        else:
            raise ValueError('Removing inexistant edge')

    def __sortedEdgesByAngle(self, previousEdge, edges):
        previousEdgeVectNormalized = numpy.array(self.node(previousEdge[1])) - numpy.array(self.node(previousEdge[0]))
        previousEdgeVectNormalized = previousEdgeVectNormalized/numpy.linalg.norm(previousEdgeVectNormalized)
        #previousEdgeVectNormalized = numpy.array((0,1))
        def angleKey(outEdge):
            edgeVectNormalized = numpy.array(self.node(outEdge[1])) - numpy.array(self.node(outEdge[0]))
            edgeVectNormalized = edgeVectNormalized/numpy.linalg.norm(edgeVectNormalized)
            return -numpy.dot(previousEdgeVectNormalized, edgeVectNormalized)

        return sorted(edges, key=angleKey)

    def dfsEdges(self):
        nodes = self.nodes()
        visitedEdges = set()
        visitedNodes = set()
        edges = {}
        dfsEdges = []

        for startNode in nodes:
            #if self.degree(startNode) != 1:
                #continue#Makes sure we don't start in the middle of a path
            stack = [startNode]
            prevEdge = None
            while stack:
                currentNode = stack[-1]
                if currentNode not in visitedNodes:
                    edges[currentNode] = self.outEdges(currentNode)
                    visitedNodes.add(currentNode)

                if edges[currentNode]:
                    if(prevEdge):
                        edges[currentNode] = self.__sortedEdgesByAngle(prevEdge, edges[currentNode])
                    edge = edges[currentNode][0]
                    if edge not in visitedEdges and (edge[1], edge[0]) not in visitedEdges:
                        visitedEdges.add(edge)
                        # Mark the traversed "to" node as to-be-explored.
                        stack.append(edge[1])
                        dfsEdges.append(edge)
                        prevEdge = edge
                    edges[currentNode].pop(0)
                else:
                    # No more edges from the current node.
                    stack.pop()
                    prevEdge = None

        return dfsEdges
"""


class OptimizePaths(inkex.Effect):
    def __init__(self):
        inkex.Effect.__init__(self)
        self.OptionParser.add_option("-t", "--tolerance",
            action="store", type="float",
            dest="tolerance", default=0.1,
            help="the distance below which 2 nodes will be merged")
        self.OptionParser.add_option("-l", "--enableLog",
                        action="store", type="inkbool",
                        dest="enableLog", default=False,
                        help="Enable logging")
        self.OptionParser.add_option("-o", "--overwriteRule",
                        action="store", type="int",
                        dest="overwriteRule", default=1,
                        help="Options to control edge overwrite rules")
        self.OptionParser.add_option("-s", "--splitSubPaths",
                        action="store", type="inkbool",
                        dest="splitSubPaths", default=False,
                        help="Split sub-paths")

    def parseSVG(self):
        vertices = []
        edges = []

        for id, node in self.selected.iteritems():
            if node.tag == inkex.addNS('path','svg'):
                d = node.get('d')
                path = simplepath.parsePath(d)
                startVertex = previousVertex = None

                for command, coords in path:
                    tcoords = tuple(coords)
                    if command == 'M':
                        vertices.append(tcoords)
                        startVertex = previousVertex = len(vertices)-1
                    elif command == 'L':
                        vertices.append(tcoords)
                        currentVertex = len(vertices)-1
                        edges.append((previousVertex, currentVertex))
                        previousVertex = currentVertex
                    elif command == 'Z':
                        edges.append((previousVertex, startVertex))
                        previousVertex = startVertex
                    elif (command == 'C' or command == 'S' or command == 'Q' or
                    command == 'T' or command == 'A'):
                        endCoords = (tcoords[-2], tcoords[-1])
                        vertices.append(endCoords)
                        currentVertex = len(vertices)-1
                        edges.append((previousVertex, currentVertex))
                        previousVertex = currentVertex
            else:
                inkex.debug("This extension only works with paths and currently doesn't support groups")

        return (vertices, edges)

    def buildGraph(self, vertices, edges):
        G = nx.Graph()
        for i, v in enumerate(vertices):
            G.add_node(i, x=v[0], y=v[1])
            self.log("N "+ str(i) + " (" + str(v[0]) + "," + str(v[1]) + ")")
        for e in edges:
            G.add_edge(e[0], e[1], weight=2)
            self.log("E "+str(e[0]) + " " + str(e[1]))
        return G

    @staticmethod
    def dist(a, b):
        return math.sqrt( (a['x'] - b['x'])**2 + (a['y'] - b['y'])**2 )

    def log(self, message):
        if(self.options.enableLog):
            inkex.debug(message)

    def mergeWithTolerance(self, G, tolerance):
        mergeTo = {}
        for ni in G.nodes():
            node_i_data = G.node[ni]
            for nj in G.nodes():
                if nj <= ni :
                    continue
                #self.log("Test " + str(ni) + " with " + str(nj))
                node_j_data = G.node[nj]
                dist_ij = self.dist(node_i_data, node_j_data)
                if (dist_ij < tolerance) and (nj not in mergeTo) and (ni not in mergeTo):
                    self.log("Merge " + str(nj) + " with " + str(ni) + " (dist="+str(dist_ij)+")")
                    mergeTo[nj] = ni

        for n in mergeTo:
            newEdges = []
            for neigh_n in G[n]:
                newEdge = None
                if neigh_n in mergeTo:
                    newEdge = (mergeTo[n], mergeTo[neigh_n])
                else:
                    newEdge = (mergeTo[n], neigh_n)

                if newEdge[0] is not newEdge[1]:
                    newEdges.append(newEdge)

            for e in newEdges:
                G.add_edge(e[0], e[1])
                #self.log("Added edge: "+str(e[0]) + " " + str(e[1]))
            G.remove_node(n)
            #self.log("Removed node: " + str(n))

    @staticmethod
    def rgbToHex(rgb):
        return '#%02x%02x%02x' % rgb

    #Color should be in hex format ("#RRGGBB"), if not specified a random color will be generated
    def addPathToInkscape(self, path, parent, color=None):
        if(color is None):
            color = colorsys.hsv_to_rgb(random.uniform(0.0, 1.0), 1.0, 1.0)
            color = tuple(x * 255 for x in color)
            color = self.rgbToHex( color )

        style = "stroke:"+color+";stroke-width:2;fill:none;"
        attribs = {'style': style, 'd': simplepath.formatPath(path) }
        inkex.etree.SubElement(parent, inkex.addNS('path','svg'), attribs )

    def removeSomeEdges(self, G, edges):
        visitedEdges = set()

        #Contains a list of [start, end] where start is the start index of a duplicate path
        #and end is the end index of the duplicate path
        edgeRangeToRemove = []
        isPrevEdgeDuplicate = False
        for i,e in enumerate(edges):
            isEdgeDuplicate = e in visitedEdges or (e[1],e[0]) in visitedEdges

            if isEdgeDuplicate:
                if not isPrevEdgeDuplicate:
                    edgeRangeToRemove.append([i, 1])
            else:
                if isPrevEdgeDuplicate:
                    edgeRangeToRemove[-1][1] = i-1

                visitedEdges.add(e)

            if isEdgeDuplicate and i == len(edges)-1:
                edgeRangeToRemove[-1][1] = i

            isPrevEdgeDuplicate = isEdgeDuplicate

        if self.options.overwriteRule == 0: #Allow overwrite
            #The last duplicate path can allways be removed
            edgeRangeToRemove = [edgeRangeToRemove[-1]] if edgeRangeToRemove else []
        elif self.options.overwriteRule == 1: #Allow overwrite except for long paths
            edgeRangeToRemove = [x for x in edgeRangeToRemove if x[1]-x[0] > 3]


        indicesToRemove = set()
        for start, end in edgeRangeToRemove:
            indicesToRemove.update(range(start, end+1))

        cleanedEdges = [e for i, e in enumerate(edges) if i not in indicesToRemove]

        return cleanedEdges

    def edgesToPaths(self, edges):
        paths = []
        path = []

        for i,e in enumerate(edges):
            #Path ends either at the last edge or when the next edge starts somewhere else
            endPath = (i == len(edges)-1 or e[1] != edges[i+1][0])

            if(not path):
                path.append(e[0])
                path.append(e[1])
            else:
                path.append(e[1])

            if endPath:
                paths.append(path)
                path = []
        return paths

    def pathsToSVG(self, G, paths):
        svgPaths = []
        for path in paths:
            svgPath = []
            for i,n in enumerate(path):
                node_i_data = G.node[n]
                command = None
                if i==0:
                    command = 'M'
                else:
                    command = 'L'
                svgPath.append([command, (node_i_data['x'], node_i_data['y'])])
            svgPaths.append(svgPath)

        if self.options.splitSubPaths:
            color = None
            parent = inkex.etree.SubElement(self.current_layer, inkex.addNS('g','svg'))
        else:
            parent = self.current_layer
            color = "#FF0000"
            svgPaths = [[x for svgPath in svgPaths for x in svgPath]]

        for svgPath in svgPaths:
            self.addPathToInkscape(svgPath, parent)

    def pathLength(self, G, path):
        length = 0.0
        for i,n in enumerate(path):
            if i > 0:
                length += self.dist(G.node[path[i-1]], G.node[path[i]])
        return length

    #Eulerization algorithm:
    #1. Find all vertices with odd valence.
    #2. Pair them up with their nearest neighbor.
    #3. Find the shortest path between each pair.
    #4. Duplicate these edges.
    def makeEulerianGraph(self, G):
        oddNodes = []
        for n in G.nodes():
            if G.degree(n) % 2 != 0:
                oddNodes.append(n)

        pathsToDuplicate = []

        while(oddNodes):
            n1 = oddNodes[0]
            shortestPaths = []
            #For every other node, find the shortest path to the closest node
            for n2 in oddNodes:
                if n2 != n1:
                    shortestPath = nx.shortest_path(G, n1, n2, 'weight')
                    shortestPaths.append(shortestPath)
                    if len(shortestPath) <= 2:
                        break #If we find a path of length 1 or 2, we assume it's good enough (to speed up calculation)
            shortestShortestPath = min(shortestPaths, key=lambda x: self.pathLength(G, x))
            closestNode = shortestShortestPath[-1]
            pathsToDuplicate.append(shortestShortestPath)
            oddNodes.pop(0)
            oddNodes.remove(closestNode)

        numberOfDuplicatedEdges = 0
        lenghtOfDuplicatedEdges = 0.0

        for path in pathsToDuplicate:
            numberOfDuplicatedEdges += len(path)-1
            pathLength = self.pathLength(G, path)
            #self.log("Path length: " + str(pathLength))
            lenghtOfDuplicatedEdges += pathLength
        #self.log("Number of duplicated edges: " + str(numberOfDuplicatedEdges))
        #self.log("Length of duplicated edges: " + str(lenghtOfDuplicatedEdges))

        G2 = nx.MultiGraph(G)
        for path in pathsToDuplicate:
            nx.add_path(G2, path)

        return G2

    def computeEdgeWeights(self,G):
        for n1,n2,key in G.edges(keys=True):
            dist = self.dist(G.node[n1], G.node[n2])
            G.add_edge(n1,n2,key,weight=dist)


    def effect(self):
        (vertices, edges) = self.parseSVG()
        G = self.buildGraph(vertices, edges)

        self.mergeWithTolerance(G, self.options.tolerance)

        """for e in G.edges():
            self.log("E "+str(e[0]) + " " + str(e[1]))
        for n in G.nodes():
            self.log("Degree of "+str(n) + ": " + str(G.degree(n)))"""
        connectedGraphs = list(nx.connected_component_subgraphs(G))
        paths = []
        for connectedGraph in connectedGraphs:
            connectedGraph = self.makeEulerianGraph(connectedGraph)
            #connectedGraph is now likely a multigraph

            self.computeEdgeWeights(connectedGraph)
            pathEdges = list(nx.eulerian_circuit(connectedGraph))
            pathEdges = self.removeSomeEdges(connectedGraph, pathEdges)
            paths.extend(self.edgesToPaths(pathEdges))

        self.log("Path number: " + str(len(paths)))
        self.log("Total path length: " + str(sum(self.pathLength(G, x) for x in paths)))

        self.pathsToSVG(G, paths)


e = OptimizePaths()
e.affect()
