/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.treernn

import com.kotlinnlp.simplednn.deeplearning.treernn.TreeEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertTrue

/**
 *
 */
class TreeEncoderSpec : Spek({

  describe("a TreeEncoder") {

    on("addNodes") {

      val treeRNN = TreeRNNUtils.buildTreeRNN()
      val treeEncoder = TreeEncoder(network = treeRNN, optimizer = null)
      val nodes: Map<Int, DenseNDArray> = TreeRNNUtils.buildNodes()

      nodes.forEach { nodeId, vector -> treeEncoder.addNode(id = nodeId, vector = vector) }

      it("should return a number of root nodes equal to the number of added nodes") {
        assertEquals(nodes.size, treeEncoder.getRootsIds().size)
      }

      it("should raise an Exception when adding a node with an id already inserted") {
        assertFails { treeEncoder.addNode(id = 1, vector = nodes[1]!!) }
      }
    }

    on("setHead") {

      val treeRNN = TreeRNNUtils.buildTreeRNN()
      val treeEncoder = TreeEncoder(network = treeRNN, optimizer = null)
      val nodes: Map<Int, DenseNDArray> = TreeRNNUtils.buildNodes()

      nodes.forEach { nodeId, vector -> treeEncoder.addNode(id = nodeId, vector = vector) }

      TreeRNNUtils.setHeads(treeEncoder)

      it("should return the expected root ids") {
        assertEquals(listOf(2, 6), treeEncoder.getRootsIds())
      }

      it("should raise an Exception when trying to set a node as head of itself") {
        assertFails { treeEncoder.setHead(3, headId = 3) }
      }

      it("should raise an Exception when trying to set the head of a not inserted node") {
        assertFails { treeEncoder.setHead(12, headId = 3) }
      }

      it("should raise an Exception when trying to set a not inserted node as head of another") {
        assertFails { treeEncoder.setHead(3, headId = 12) }
      }

      it("should create the expected encoding of the node 1") {
        assertTrue {
          treeEncoder.getNode(1).encoding.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.26221, -0.021976)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should create the expected encoding of the node 2") {
        assertTrue {
          treeEncoder.getNode(2).encoding.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.496212, 0.012471)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should create the expected encoding of the node 3") {
        assertTrue {
          treeEncoder.getNode(3).encoding.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.910216, -0.902148)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should create the expected encoding of the node 4") {
        assertTrue {
          treeEncoder.getNode(4).encoding.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.190158, -0.096582)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should create the expected encoding of the node 5") {
        assertTrue {
          treeEncoder.getNode(5).encoding.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.760148, -0.493145)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should create the expected encoding of the node 6") {
        assertTrue {
          treeEncoder.getNode(6).encoding.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.6779, -0.650755)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should create the expected encoding of the node 7") {
        assertTrue {
          treeEncoder.getNode(7).encoding.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.9403, -0.771791)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should create the expected encoding of the node 8") {
        assertTrue {
          treeEncoder.getNode(8).encoding.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.968959, -0.843283)),
            tolerance = 1.0e-06
          )
        }
      }
    }

    on("setHead with a different order") {

      val treeRNN = TreeRNNUtils.buildTreeRNN()
      val treeEncoder1 = TreeEncoder(network = treeRNN, optimizer = null)
      val treeEncoder2 = TreeEncoder(network = treeRNN, optimizer = null)
      val nodes: Map<Int, DenseNDArray> = TreeRNNUtils.buildNodes()

      nodes.forEach { nodeId, vector -> treeEncoder1.addNode(id = nodeId, vector = vector) }
      nodes.forEach { nodeId, vector -> treeEncoder2.addNode(id = nodeId, vector = vector) }

      TreeRNNUtils.setHeads(treeEncoder1)
      TreeRNNUtils.setHeads2(treeEncoder2)

      it("should set the same root nodes") {
        val rootIds1 = treeEncoder1.getRootsIds().sorted()
        val rootIds2 = treeEncoder2.getRootsIds().sorted()

        assertEquals(rootIds1, rootIds2)
      }

      it("should calculate the same encoding for the node 1") {
        assertTrue { treeEncoder1.getNode(1).encoding.equals(treeEncoder2.getNode(1).encoding, tolerance = 1.0e-06) }
      }

      it("should calculate the same encoding for the node 2") {
        assertTrue { treeEncoder1.getNode(2).encoding.equals(treeEncoder2.getNode(2).encoding, tolerance = 1.0e-06) }
      }

      it("should calculate the same encoding for the node 3") {
        assertTrue { treeEncoder1.getNode(3).encoding.equals(treeEncoder2.getNode(3).encoding, tolerance = 1.0e-06) }
      }

      it("should calculate the same encoding for the node 4") {
        assertTrue { treeEncoder1.getNode(4).encoding.equals(treeEncoder2.getNode(4).encoding, tolerance = 1.0e-06) }
      }

      it("should calculate the same encoding for the node 5") {
        assertTrue { treeEncoder1.getNode(5).encoding.equals(treeEncoder2.getNode(5).encoding, tolerance = 1.0e-06) }
      }

      it("should calculate the same encoding for the node 6") {
        assertTrue { treeEncoder1.getNode(6).encoding.equals(treeEncoder2.getNode(6).encoding, tolerance = 1.0e-06) }
      }

      it("should calculate the same encoding for the node 7") {
        assertTrue { treeEncoder1.getNode(7).encoding.equals(treeEncoder2.getNode(7).encoding, tolerance = 1.0e-06) }
      }

      it("should calculate the same encoding for the node 8") {
        assertTrue { treeEncoder1.getNode(8).encoding.equals(treeEncoder2.getNode(8).encoding, tolerance = 1.0e-06) }
      }
    }

    on("getNode") {

      val treeRNN = TreeRNNUtils.buildTreeRNN()
      val treeEncoder = TreeEncoder(network = treeRNN, optimizer = null)
      val nodes: Map<Int, DenseNDArray> = TreeRNNUtils.buildNodes()

      nodes.forEach { nodeId, vector -> treeEncoder.addNode(id = nodeId, vector = vector) }

      val extractedNode = treeEncoder.getNode(4)

      it("should raise an Exception when trying to get a node not previously inserted") {
        assertFails { treeEncoder.getNode(12) }
      }

      it("should return a node with the expected id") {
        assertEquals(4, extractedNode.id)
      }

      it("should return a node with the expected vector") {
        assertEquals(nodes[4]!!, extractedNode.vector)
      }
    }

    on("setEncodingErrors") {

      val treeRNN = TreeRNNUtils.buildTreeRNN()
      val treeEncoder = TreeEncoder(network = treeRNN, optimizer = null)
      val nodes: Map<Int, DenseNDArray> = TreeRNNUtils.buildNodes()
      val encodingErrors: Map<Int, DenseNDArray> = TreeRNNUtils.getEncodingErrors()

      nodes.forEach { nodeId, vector -> treeEncoder.addNode(id = nodeId, vector = vector) }

      TreeRNNUtils.setHeads(treeEncoder)

      encodingErrors.forEach { nodeId, errors -> treeEncoder.setEncodingErrors(nodeId = nodeId, errors = errors) }

      it("should raise an Exception when trying to set the encoding errors of a node not previously inserted") {
        assertFails { treeEncoder.setEncodingErrors(12, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2))) }
      }

      it("should raise an Exception when trying to set not compatible encoding errors") {
        assertFails { treeEncoder.setEncodingErrors(3, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3))) }
      }

      it("should raise an Exception when trying to set the encoding errors of a not-root node") {
        assertFails { treeEncoder.setEncodingErrors(3, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2))) }
      }

      it("should set the expected errors into the first root node") {
        assertTrue { encodingErrors[2]!!.equals(treeEncoder.getNode(2).getNodeErrors()!!, tolerance = 1.0e-08) }
      }

      it("should set the expected errors into the second root node") {
        assertTrue { encodingErrors[6]!!.equals(treeEncoder.getNode(6).getNodeErrors()!!, tolerance = 1.0e-08) }
      }
    }
  }
})
