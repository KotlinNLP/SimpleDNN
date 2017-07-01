/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionlayer.AttentionLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import deeplearning.utils.AttentionLayerUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
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

      it("should raise an Exception with an empty input sequence") {
        assertFails {
          AttentionLayerStructure(
            inputSequence = ArrayList<AugmentedArray<DenseNDArray>>(),
            attentionSequence = attentionSequence)
        }
      }

      it("should raise an Exception with an empty attention sequence") {
        assertFails {
          AttentionLayerStructure(
            inputSequence = inputSequence,
            attentionSequence = ArrayList<DenseNDArray>())
        }
      }

      it("should raise an Exception with input and attention sequences not compatible") {

        val wrongAttentionSequence = AttentionLayerUtils.buildAttentionSequence(inputSequence)
        wrongAttentionSequence.removeAt(1)

        assertFails {
          AttentionLayerStructure(inputSequence = inputSequence, attentionSequence = wrongAttentionSequence)
        }
      }
    }

    context("correct initialization") {

      val inputSequence = AttentionLayerUtils.buildInputSequence()
      val attentionSequence = AttentionLayerUtils.buildAttentionSequence(inputSequence)
      val structure = AttentionLayerStructure(inputSequence = inputSequence, attentionSequence = attentionSequence)

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
  }
})
