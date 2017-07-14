/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning

import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.treernn.TreeEncoder
import com.kotlinnlp.simplednn.deeplearning.treernn.TreeRNN
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class TreeRNNSpec : Spek({

  describe("a TreeEncoder") {

    val inputLayerSize = 2

    val treeRNN = TreeRNN(
      inputLayerSize = inputLayerSize,
      hiddenLayerSize = 5,
      hiddenLayerConnectionType = LayerType.Connection.GRU).initialize()

    val treeEncoder = TreeEncoder(network = treeRNN, optimizer = null)

    /**
     * Id -> Head
     */
    val sequence = listOf(0, 1, 2, 3, 4, 5)

    on("addNode") {
      sequence.forEach {
        treeEncoder.addNode(id = it, vector = DenseNDArrayFactory.zeros(Shape(inputLayerSize)))
      }

      it("should return the expected number of root nodes") {
        assertEquals(sequence.size, treeEncoder.getRootsIds().size)
      }
    }

  }

})
