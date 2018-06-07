/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.treernn

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.core.layers.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
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
      val treeEncoder = TreeEncoder(network = treeRNN)
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
      val treeEncoder = TreeEncoder(network = treeRNN)
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
      val treeEncoder1 = TreeEncoder(network = treeRNN)
      val treeEncoder2 = TreeEncoder(network = treeRNN)
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
      val treeEncoder = TreeEncoder(network = treeRNN)
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

    on("addEncodingErrors") {

      val treeRNN = TreeRNNUtils.buildTreeRNN()
      val treeEncoder = TreeEncoder(network = treeRNN)
      val nodes: Map<Int, DenseNDArray> = TreeRNNUtils.buildNodes()

      nodes.forEach { nodeId, vector -> treeEncoder.addNode(id = nodeId, vector = vector) }

      TreeRNNUtils.setHeads(treeEncoder)

      it("should raise an Exception when trying to set the encoding errors of a node not previously inserted") {
        assertFails { treeEncoder.addEncodingErrors(12, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2))) }
      }

      it("should raise an Exception when trying to set not compatible encoding errors") {
        assertFails { treeEncoder.addEncodingErrors(3, DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2, 0.3))) }
      }
    }

    on("propagateErrors") {

      val treeRNN = TreeRNNUtils.buildTreeRNN()
      val treeEncoder = TreeEncoder(network = treeRNN)
      val nodes: Map<Int, DenseNDArray> = TreeRNNUtils.buildNodes()
      val encodingErrors: Map<Int, DenseNDArray> = TreeRNNUtils.getEncodingErrors()

      nodes.forEach { nodeId, vector -> treeEncoder.addNode(id = nodeId, vector = vector) }

      TreeRNNUtils.setHeads(treeEncoder)

      encodingErrors.forEach { nodeId, errors -> treeEncoder.addEncodingErrors(nodeId = nodeId, errors = errors) }

      val optimizer = ParamsOptimizer(params = treeRNN.model, updateMethod = LearningRateMethod(learningRate = 0.01))
      treeEncoder.propagateErrors(optimizer)

      it("should match the expected vector errors of the node 1") {
        assertTrue {
          treeEncoder.getNode(1).vectorErrors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.057933, -0.20179)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected vector errors of the node 2") {
        assertTrue {
          treeEncoder.getNode(2).vectorErrors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.14089, 0.20105)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected vector errors of the node 3") {
        assertTrue {
          treeEncoder.getNode(3).vectorErrors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.025417, -0.068414)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should match the expected vector errors of the node 4") {
        assertTrue {
          treeEncoder.getNode(4).vectorErrors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.009037, -0.076462)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should match the expected vector errors of the node 5") {
        assertTrue {
          treeEncoder.getNode(5).vectorErrors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.010593, -0.009837)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should match the expected vector errors of the node 6") {
        assertTrue {
          treeEncoder.getNode(6).vectorErrors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.231024, -0.32538)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should match the expected vector errors of the node 7") {
        assertTrue {
          treeEncoder.getNode(7).vectorErrors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.165773, -0.053794)),
            tolerance = 1.0e-06
          )
        }
      }

      it("should match the expected vector errors of the node 8") {
        assertTrue {
          treeEncoder.getNode(8).vectorErrors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.002945, -0.056789)),
            tolerance = 1.0e-06
          )
        }
      }
    }

    on("update") {

      val treeRNN = TreeRNNUtils.buildTreeRNN()
      val treeEncoder = TreeEncoder(network = treeRNN)
      val nodes: Map<Int, DenseNDArray> = TreeRNNUtils.buildNodes()
      val encodingErrors: Map<Int, DenseNDArray> = TreeRNNUtils.getEncodingErrors()

      val optimizer = ParamsOptimizer(params = treeRNN.model, updateMethod = LearningRateMethod(learningRate = 0.1))

      nodes.forEach { nodeId, vector -> treeEncoder.addNode(id = nodeId, vector = vector) }

      TreeRNNUtils.setHeads(treeEncoder)

      encodingErrors.forEach { nodeId, errors -> treeEncoder.addEncodingErrors(nodeId = nodeId, errors = errors) }

      treeEncoder.propagateErrors(optimizer)
      optimizer.update()

      val concatParams = treeRNN.concatNetwork.model.paramsPerLayer[0] as FeedforwardLayerParameters
      val leftRNNParams = treeRNN.leftRNN.model.paramsPerLayer[0] as SimpleRecurrentLayerParameters
      val rightRNNParams = treeRNN.rightRNN.model.paramsPerLayer[0] as SimpleRecurrentLayerParameters

      it("should match the expected updated values of the concat network biases") {
        assertTrue {
           concatParams.unit.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.903129, -0.809295)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected updated values of the concat network weights") {
        assertTrue {
          (concatParams.unit.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.19561, -0.201012, -0.400524, 1.003945, -0.501027, -0.398002),
              doubleArrayOf(0.501749, 0.494214, 0.202465, -0.800209, 0.492504, 0.091253)
            )),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected updated values of the leftRNN biases") {
        assertTrue {
           leftRNNParams.unit.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.79655, 0.195259, -0.303113)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected updated values of the leftRNN weights") {
        assertTrue {
          (leftRNNParams.unit.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.602513, 0.802564),
              doubleArrayOf(-0.295092, 0.005574),
              doubleArrayOf(0.902109, -0.796982)
            )),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected updated values of the leftRNN recurrent weights") {
        assertTrue {
          leftRNNParams.unit.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.100383, 0.699583, 0.000294),
              doubleArrayOf(0.200129, 0.899859, -0.199901),
              doubleArrayOf(-0.500351, -0.199617, -0.400269)
            )),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected updated values of the rightRNN biases") {
        assertTrue {
           rightRNNParams.unit.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.708738, 0.898502, 0.401487)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected updated values of the rightRNN weights") {
        assertTrue {
          (rightRNNParams.unit.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.097327, 0.894518),
              doubleArrayOf(-1.000919, -0.398736),
              doubleArrayOf(0.396431, -0.801835)
            )),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected updated values of the rightRNN recurrent weights") {
        assertTrue {
          rightRNNParams.unit.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(1.000556, -0.09775, 0.700684),
              doubleArrayOf(-0.700047, 0.798187, -1.000972),
              doubleArrayOf(-0.000055, 0.800068, 0.000076)
            )),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
