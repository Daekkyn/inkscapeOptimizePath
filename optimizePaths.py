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
import networkx as nx

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

    def parseSVG(self):
        vertices = []
        edges = []

        for id, node in self.selected.iteritems():
            if node.tag == inkex.addNS('path','svg'):
                d = node.get('d')
                path = simplepath.parsePath(d)
                #start = prev = None
                startVertex = previousVertex = None

                for command, coords in path:
                    tcoords = tuple(coords)
                    #self.log(command + " " + str(tcoords))
                    if command == 'M':
                        #start = prev = tcoords
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
                        self.log("C: " + str(tcoords))
                        endCoords = (tcoords[-2], tcoords[-1])
                        vertices.append(endCoords)
                        currentVertex = len(vertices)-1
                        edges.append((previousVertex, currentVertex))
                        previousVertex = currentVertex

            #elif node.tag == inkex.addNS('polygon','svg'):

        #for v in vertices:
            #self.log("Vertex: " + str(v))
        #for e in edges:
            #self.log("Edge: " + str(e))

        return (vertices, edges)

    def buildGraph(self, vertices, edges):
        G = nx.Graph()
        for i, v in enumerate(vertices):
            G.add_node(i, x=v[0], y=v[1])
            self.log("N "+ str(i) + " (" + str(v[0]) + "," + str(v[1]) + ")")
        for e in edges:
            G.add_edge(e[0], e[1])
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
            node_i = G.node[ni]
            for nj in G.nodes():
                if nj <= ni :
                    continue
                #self.log("Test " + str(ni) + " with " + str(nj))
                node_j = G.node[nj]
                dist_ij = self.dist(node_i, node_j)
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
                self.log("Added edge: "+str(e[0]) + " " + str(e[1]))
            G.remove_node(n)
            self.log("Removed node: " + str(n))


    def graphToSVG(self, G):
        parent = self.current_layer
        style = {'stroke': '#FF0000','stroke-width': '2', 'fill': 'none'}
        prevEdge = None
        path = []

        for e in nx.edge_dfs(G):
            node_i = G.node[e[0]]
            node_j = G.node[e[1]]

            if (prevEdge == None or prevEdge[1] != e[0]):
                path.append(['M', (node_i['x'], node_i['y'])])
                path.append(['L', (node_j['x'], node_j['y'])])
            else:
                path.append(['L', (node_j['x'], node_j['y'])])

            prevEdge = e

        attribs = {'style': simplestyle.formatStyle(style),
                    'd': simplepath.formatPath(path) }
        inkex.etree.SubElement(parent, inkex.addNS('path','svg'), attribs )

    def effect(self):
        (vertices, edges) = self.parseSVG()
        G = self.buildGraph(vertices, edges)
        self.log("Number of edges: "+str(len(edges)))

        self.mergeWithTolerance(G, self.options.tolerance)
        self.log("Number of edges after cleaning: "+str(G.number_of_edges()))

        for e in G.edges():
            self.log("E "+str(e[0]) + " " + str(e[1]))
        for n in G.nodes():
            self.log("Degree of "+str(n) + ": " + str(G.degree(n)))

        self.graphToSVG(G)

e = OptimizePaths()
e.affect()
