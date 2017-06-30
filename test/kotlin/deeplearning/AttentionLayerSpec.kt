/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning

import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionlayer.AttentionLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import deeplearning.utils.AttentionLayerUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class AttentionLayerSpec : Spek({

  describe("an AttentionLayer") {

    on("forward") {

      val attentionLayer = AttentionLayerUtils.buildAttentionLayer()
      val inputSequence = AttentionLayerUtils.buildInputSequence()
      val structure = AttentionLayerStructure(
        inputSize = 4,
        inputSequence = inputSequence,
        attentionSequence = AttentionLayerUtils.buildAttentionSequence(inputSequence)
      )

      attentionLayer.forward(structure)

      it("should match the expected attention context") {
        assertTrue {
          structure.attentionContext.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.093476, 0.040542, 0.039525)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected importance score") {
        assertTrue {
          structure.importanceScore.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.304352, 0.348001, 0.347647)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected output array") {
        assertTrue {
          structure.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.191447, 0.282824, 0.030317, 0.530541)),
            tolerance = 1.0e-06)
        }
      }
    }

    on("backward") {

      val attentionLayer = AttentionLayerUtils.buildAttentionLayer()
      val inputSequence = AttentionLayerUtils.buildInputSequence()
      val structure = AttentionLayerStructure(
        inputSize = 4,
        inputSequence = inputSequence,
        attentionSequence = AttentionLayerUtils.buildAttentionSequence(inputSequence)
      )

      attentionLayer.forward(structure)

      structure.outputArray.assignErrors(AttentionLayerUtils.buildOutputErrors())
      attentionLayer.backward(structure, propagateToInput = true)

      it("should match the expected errors of the context vector") {
        assertTrue {
          structure.contextVectorErrors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.02985, 0.006539)),
            tolerance = 1.0e-06)
        }
      }

      val attentionErrors: Array<DenseNDArray> = structure.getAttentionErrors()

      it("should match the expected errors of the first attention array") {
        assertTrue {
          attentionErrors[0].equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.027623, -0.046039)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the second attention array") {
        assertTrue {
          attentionErrors[1].equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.006529, -0.010882)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the third attention array") {
        assertTrue {
          attentionErrors[2].equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.034152, 0.056921)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the first input") {
        assertTrue {
          structure.inputSequence[0].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.06087, 0.152176, 0.030435, -0.152176)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the second input") {
        assertTrue {
          structure.inputSequence[1].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.0696, 0.174, 0.0348, -0.174)),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the third input") {
        assertTrue {
          structure.inputSequence[2].errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.069529, 0.173823, 0.034765, -0.173823)),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
