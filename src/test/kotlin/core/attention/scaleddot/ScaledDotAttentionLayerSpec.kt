/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.attention.scaleddot

import com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot.ScaledDotAttentionLayer
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertFails
import kotlin.test.assertTrue

/**
 *
 */
class ScaledDotAttentionLayerSpec : Spek({

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
            doubleArrayOf(0.83, 0.7, -0.25, -0.58),
            doubleArrayOf(0.0, 0.2, 0.57, -2.08),
            doubleArrayOf(-0.07, 0.0, 0.29, 0.5)
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
            doubleArrayOf(0.312291, 0.347165, 0.170855, -0.813202),
            doubleArrayOf(0.232861, 0.284047, 0.21555, -0.694914),
            doubleArrayOf(0.236194, 0.28672, 0.21373, -0.700304)
          )).equals(
            DenseNDArrayFactory.fromRows(layer.outputArrays.map { it.values }),
            tolerance = 1.0e-06)
        }
      }
    }

    context("backward") {

      val inputSequence = ScaledDotAttentionLayerUtils.buildInputSequence()
      val layer = ScaledDotAttentionLayer(
        inputArrays = inputSequence,
        params = ScaledDotAttentionLayerUtils.buildAttentionParams()
      )

      layer.forward()

      layer.outputArrays.zip(ScaledDotAttentionLayerUtils.buildOutputErrors()).forEach { (array, errors) ->
        array.assignErrors(errors)
      }

      val paramsErrors: ParamsErrorsList = layer.backward(propagateToInput = true)

      it("should match the expected errors of the queries") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.291064, 0.090078),
            doubleArrayOf(-0.214319, -0.065291),
            doubleArrayOf(0.084357, 0.057063)
          )).equals(layer.queries.errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the keys") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.06886, -0.025612),
            doubleArrayOf(-0.039958, 0.089393),
            doubleArrayOf(-0.028902, -0.063781)
          )).equals(layer.keys.errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.15834, -0.431875, -0.371149, -0.450847),
            doubleArrayOf(-0.22708, -0.436103, -0.339456, -0.438166),
            doubleArrayOf(-0.31458, -0.432022, -0.289395, -0.410987)
          )).equals(layer.values.errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the queries weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.016041, 0.011542, 0.411981, 0.020054),
            doubleArrayOf(0.013733, -0.011181, 0.126774, 0.013227)
          )).equals(paramsErrors.getErrorsOf(layer.params.queries)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the keys weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.016236, 0.034682, 0.089944, 0.003569),
            doubleArrayOf(-0.053586, 0.076537, -0.085626, 0.043391)
          )).equals(paramsErrors.getErrorsOf(layer.params.keys)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the values weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.082502, -0.112504, 0.016450, -0.339584),
            doubleArrayOf(0.261195, -0.391573, -0.083416, -0.694412),
            doubleArrayOf(0.231369, -0.352726, -0.096414, -0.552133),
            doubleArrayOf(0.276126, -0.416815, -0.099046, -0.703238)
          )).equals(paramsErrors.getErrorsOf(layer.params.values)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the first input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.065071, 0.7222, -0.355602, -0.06106)).equals(
            layer.inputArrays[0].errors,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the second input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.141053, 0.454059, -0.576099, -0.221435)).equals(
            layer.inputArrays[1].errors,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the third input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.149318, 0.319997, -0.537894, -0.120615)).equals(
            layer.inputArrays[2].errors,
            tolerance = 1.0e-06)
        }
      }
    }

    context("forward with dropout") {

      val inputSequence = ScaledDotAttentionLayerUtils.buildInputSequence()
      val layer = ScaledDotAttentionLayer(
        inputArrays = inputSequence,
        attentionDropout = 1.0e-12, // activate the dropout actually without dropping (very low probability)
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
            doubleArrayOf(0.83, 0.7, -0.25, -0.58),
            doubleArrayOf(0.0, 0.2, 0.57, -2.08),
            doubleArrayOf(-0.07, 0.0, 0.29, 0.5)
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
            doubleArrayOf(0.312291, 0.347165, 0.170855, -0.813202),
            doubleArrayOf(0.232861, 0.284047, 0.21555, -0.694914),
            doubleArrayOf(0.236194, 0.28672, 0.21373, -0.700304)
          )).equals(
            DenseNDArrayFactory.fromRows(layer.outputArrays.map { it.values }),
            tolerance = 1.0e-06)
        }
      }
    }

    context("backward with dropout") {

      val inputSequence = ScaledDotAttentionLayerUtils.buildInputSequence()
      val layer = ScaledDotAttentionLayer(
        inputArrays = inputSequence,
        attentionDropout = 1.0e-12, // activate the dropout actually without dropping (very low probability)
        params = ScaledDotAttentionLayerUtils.buildAttentionParams()
      )

      layer.forward()

      layer.outputArrays.zip(ScaledDotAttentionLayerUtils.buildOutputErrors()).forEach { (array, errors) ->
        array.assignErrors(errors)
      }

      val paramsErrors: ParamsErrorsList = layer.backward(propagateToInput = true)

      it("should match the expected errors of the queries") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.291064, 0.090078),
            doubleArrayOf(-0.214319, -0.065291),
            doubleArrayOf(0.084357, 0.057063)
          )).equals(layer.queries.errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the keys") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.06886, -0.025612),
            doubleArrayOf(-0.039958, 0.089393),
            doubleArrayOf(-0.028902, -0.063781)
          )).equals(layer.keys.errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.15834, -0.431875, -0.371149, -0.450847),
            doubleArrayOf(-0.22708, -0.436103, -0.339456, -0.438166),
            doubleArrayOf(-0.31458, -0.432022, -0.289395, -0.410987)
          )).equals(layer.values.errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the queries weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.016041, 0.011542, 0.411981, 0.020054),
            doubleArrayOf(0.013733, -0.011181, 0.126774, 0.013227)
          )).equals(paramsErrors.getErrorsOf(layer.params.queries)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the keys weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.016236, 0.034682, 0.089944, 0.003569),
            doubleArrayOf(-0.053586, 0.076537, -0.085626, 0.043391)
          )).equals(paramsErrors.getErrorsOf(layer.params.keys)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the values weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.082502, -0.112504, 0.016450, -0.339584),
            doubleArrayOf(0.261195, -0.391573, -0.083416, -0.694412),
            doubleArrayOf(0.231369, -0.352726, -0.096414, -0.552133),
            doubleArrayOf(0.276126, -0.416815, -0.099046, -0.703238)
          )).equals(paramsErrors.getErrorsOf(layer.params.values)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the first input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.065071, 0.7222, -0.355602, -0.06106)).equals(
            layer.inputArrays[0].errors,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the second input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.141053, 0.454059, -0.576099, -0.221435)).equals(
            layer.inputArrays[1].errors,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the third input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.149318, 0.319997, -0.537894, -0.120615)).equals(
            layer.inputArrays[2].errors,
            tolerance = 1.0e-06)
        }
      }
    }
  }
})
