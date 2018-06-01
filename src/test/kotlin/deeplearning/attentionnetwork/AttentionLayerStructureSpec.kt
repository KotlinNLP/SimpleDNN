/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.attentionnetwork

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.attentionlayer.AttentionParameters
import com.kotlinnlp.simplednn.core.attentionlayer.AttentionLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import deeplearning.attentionnetwork.utils.AttentionLayerUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertFails
import kotlin.test.assertTrue

/**
 *
 */
class AttentionLayerStructureSpec : Spek({

  describe("an AttentionLayerStructure") {

    context("wrong initialization") {

      val inputSequence = AttentionLayerUtils.buildInputSequence()
      val attentionSequence = AttentionLayerUtils.buildAttentionSequence(inputSequence)
      val params = AttentionLayerUtils.buildAttentionParams()

      it("should raise an Exception with an empty input sequence") {
        assertFails {
          AttentionLayerStructure(
            inputSequence = ArrayList<AugmentedArray<DenseNDArray>>(),
            attentionSequence = attentionSequence,
            params = params)
        }
      }

      it("should raise an Exception with an empty attention sequence") {
        assertFails {
          AttentionLayerStructure(
            inputSequence = inputSequence,
            attentionSequence = ArrayList(),
            params = params)
        }
      }

      it("should raise an Exception with input and attention sequences not compatible") {

        val wrongAttentionSequence = AttentionLayerUtils.buildAttentionSequence(inputSequence)
        wrongAttentionSequence.removeAt(1)

        assertFails {
          AttentionLayerStructure(
            inputSequence = inputSequence,
            attentionSequence = wrongAttentionSequence,
            params = params)
        }
      }

      it("should raise an Exception with a attention arrays with a not expected size") {

        val wrongAttentionSequence = AttentionLayerUtils.buildAttentionSequence(inputSequence)
        wrongAttentionSequence.removeAt(1)
        wrongAttentionSequence.add(DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 0.1, 0.3)))

        assertFails {
          AttentionLayerStructure(
            inputSequence = inputSequence,
            attentionSequence = wrongAttentionSequence,
            params = params)
        }
      }
    }

    context("correct initialization") {

      val inputSequence = AttentionLayerUtils.buildInputSequence()
      val attentionSequence = AttentionLayerUtils.buildAttentionSequence(inputSequence)
      val structure = AttentionLayerStructure(
        inputSequence = inputSequence,
        attentionSequence = attentionSequence,
        params = AttentionLayerUtils.buildAttentionParams())

      it("should initialize the attention matrix correctly") {
        assertTrue {
          structure.attentionMatrix.values.equals(
            DenseNDArrayFactory.arrayOf(arrayOf(
              attentionSequence[0].toDoubleArray(),
              attentionSequence[1].toDoubleArray(),
              attentionSequence[2].toDoubleArray()
            )),
            tolerance = 1.0e-06
          )
        }
      }

      structure.attentionMatrix.assignErrors(DenseNDArrayFactory.arrayOf(arrayOf(
        doubleArrayOf(0.1, 0.2),
        doubleArrayOf(0.3, 0.4),
        doubleArrayOf(0.5, 0.6)
      )))

      val attentionErrors = structure.getAttentionErrors()

      it("should match the expected errors of the first attention array") {
        assertTrue {
          attentionErrors[0].equals(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.2)), tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the second attention array") {
        assertTrue {
          attentionErrors[1].equals(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.3, 0.4)), tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the third attention array") {
        assertTrue {
          attentionErrors[2].equals(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, 0.6)), tolerance = 1.0e-06)
        }
      }
    }

    on("forward") {

      val inputSequence = AttentionLayerUtils.buildInputSequence()
      val structure = AttentionLayerStructure(
        inputSequence = inputSequence,
        attentionSequence = AttentionLayerUtils.buildAttentionSequence(inputSequence),
        params = AttentionLayerUtils.buildAttentionParams()
      )

      structure.forward()

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

      val inputSequence = AttentionLayerUtils.buildInputSequence()
      val structure = AttentionLayerStructure(
        inputSequence = inputSequence,
        attentionSequence = AttentionLayerUtils.buildAttentionSequence(inputSequence),
        params = AttentionLayerUtils.buildAttentionParams()
      )

      structure.forward()

      structure.outputArray.assignErrors(AttentionLayerUtils.buildOutputErrors())

      val errors = AttentionParameters(attentionSize = 2)
      structure.backward(paramsErrors = errors, propagateToInput = true)

      it("should match the expected errors of the context vector") {
        assertTrue {
          errors.contextVector.values.equals(
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
