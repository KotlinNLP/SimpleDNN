/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.attention

import com.kotlinnlp.simplednn.deeplearning.attention.attentionnetwork.AttentionNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import core.attention.AttentionLayerUtils
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class AttentionNetworkSpec : Spek({

  describe("an AttentionNetwork") {

    context("forward") {

      val network: AttentionNetwork<DenseNDArray> = AttentionNetworkUtils.buildNetwork()
      val output = network.forward(inputSequence = AttentionLayerUtils.buildInputSequence())

      it("should match the expected output") {
        assertTrue {
          output.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.191447, 0.282824, 0.030317, 0.530541)),
            tolerance = 1.0e-06)
        }
      }
    }

    context("forward with external attention arrays") {

      val network: AttentionNetwork<DenseNDArray> = AttentionNetworkUtils.buildNetwork()
      val inputSequence = AttentionLayerUtils.buildInputSequence()
      val output = network.forward(
        inputSequence = inputSequence,
        attentionSequence = AttentionLayerUtils.buildAttentionSequence(inputSequence))

      it("should match the expected output") {
        assertTrue {
          output.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.191447, 0.282824, 0.030317, 0.530541)),
            tolerance = 1.0e-06)
        }
      }
    }

    context("backward") {

      val network: AttentionNetwork<DenseNDArray> = AttentionNetworkUtils.buildNetwork()

      network.forward(inputSequence = AttentionLayerUtils.buildInputSequence())
      network.backward(outputErrors = AttentionLayerUtils.buildOutputErrors())

      val inputErrors = network.getInputErrors()

      it("should match the expected errors of the first input") {
        assertTrue {
          inputErrors[0].equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.076383, 0.148207, 0.021613, -0.175261)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the second input") {
        assertTrue {
          inputErrors[1].equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.073648, 0.172493, 0.032455, -0.179081)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the third input") {
        assertTrue {
          inputErrors[2].equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.048449, 0.181651, 0.046974, -0.147318)),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
