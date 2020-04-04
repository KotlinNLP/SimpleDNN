/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.attention.scaleddot

import com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot.ScaledDotAttentionLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertFails
import kotlin.test.assertTrue

/**
 *
 */
class ScaledDotAttentionLayerStructureSpec : Spek({

  describe("a ScaledDotAttentionLayer") {

    context("wrong initialization") {

      it("should raise an Exception with an empty input sequence") {
        assertFails {
          ScaledDotAttentionLayer(
            inputArrays = mutableListOf(),
            params = ScaledDotAttentionLayerUtils.buildAttentionParams())
        }
      }
    }

    context("forward") {

      val inputSequence = ScaledDotAttentionLayerUtils.buildInputSequence()
      val layer = ScaledDotAttentionLayer(
        inputArrays = inputSequence,
        params = ScaledDotAttentionLayerUtils.buildAttentionParams()
      )

      layer.forward()

      it("should match the expected queries") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.22, 0.3),
            doubleArrayOf(-0.17, 0.24),
            doubleArrayOf(-0.15, 0.23)
          )).equals(
            layer.queries.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected keys") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(1.66, 0.12),
            doubleArrayOf(0.88, -0.02),
            doubleArrayOf(-0.3, -0.46)
          )).equals(
            layer.keys.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.83, 0.7, -0.25),
            doubleArrayOf(0.0, 0.2, 0.57),
            doubleArrayOf(-0.07, 0.0, 0.29)
          )).equals(
            layer.values.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected attention scores") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.398142, 0.342329, 0.259529),
            doubleArrayOf(0.310603, 0.333125, 0.356272),
            doubleArrayOf(0.314262, 0.333682, 0.352055)
          )).equals(
            DenseNDArrayFactory.fromRows(layer.attentionAct),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected output arrays") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.312291, 0.347165, 0.170855),
            doubleArrayOf(0.232861, 0.284047, 0.21555),
            doubleArrayOf(0.236194, 0.28672, 0.21373)
          )).equals(
            DenseNDArrayFactory.fromRows(layer.outputArrays.map { it.values }),
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
