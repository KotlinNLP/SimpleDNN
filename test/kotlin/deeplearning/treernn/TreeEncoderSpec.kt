/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.treernn

import com.kotlinnlp.simplednn.deeplearning.treernn.TreeEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals

/**
 *
 */
class TreeEncoderSpec : Spek({

  describe("a TreeEncoder") {

    val treeRNN = TreeRNNUtils.buildTreeRNN()

    val treeEncoder = TreeEncoder(network = treeRNN, optimizer = null)

    val nodes: Map<Int, DenseNDArray> = TreeRNNUtils.buildNodes()

    on("addNode") {

      nodes.forEach { node_id, vector -> treeEncoder.addNode(id = node_id, vector = vector) }

      it("should return the expected number of initial root nodes") {
        assertEquals(nodes.size, treeEncoder.getRootsIds().size)
      }
    }
  }
})
