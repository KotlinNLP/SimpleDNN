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
            doubleArrayOf(0.92, 1.1),
            doubleArrayOf(0.53, 1.04),
            doubleArrayOf(0.55, 1.03)
          )).equals(
            layer.queries.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected keys") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.96, 0.02),
            doubleArrayOf(0.18, -0.12),
            doubleArrayOf(-1.0, -0.56)
          )).equals(
            layer.keys.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.63, 0.5, -0.95, 0.32),
            doubleArrayOf(-0.2, 0.0, -0.13, -1.18),
            doubleArrayOf(-0.27, -0.2, -0.41, 1.4)
          )).equals(
            layer.values.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected attention scores") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.582109, 0.314298, 0.103593),
            doubleArrayOf(0.503361, 0.339012, 0.157628),
            doubleArrayOf(0.506943, 0.338013, 0.155044)
          )).equals(
            DenseNDArrayFactory.fromRows(layer.attentionAct),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected output arrays") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.275899, 0.270336, -0.636335, -0.039567),
            doubleArrayOf(0.206755, 0.220155, -0.586891, -0.018279),
            doubleArrayOf(0.209909, 0.222462, -0.589105, -0.019572)
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
            doubleArrayOf(0.166976, 0.04868),
            doubleArrayOf(-0.157867, -0.044738),
            doubleArrayOf(-0.118836, -0.00275)
          )).equals(layer.queries.errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the keys") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.118337, -0.282144),
            doubleArrayOf(0.200448, 0.381435),
            doubleArrayOf(-0.082111, -0.099291)
          )).equals(layer.keys.errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.299378, -0.679784, -0.557768, -0.696967),
            doubleArrayOf(-0.254008, -0.432802, -0.321912, -0.42746),
            doubleArrayOf(-0.146614, -0.187414, -0.12032, -0.175574)
          )).equals(layer.values.errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the queries weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.023508, 0.065794, 0.260786, -0.049875),
            doubleArrayOf(0.002072, 0.004134, 0.075128, -0.007132)
          )).equals(paramsErrors.getErrorsOf(layer.params.queries.weights)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the keys weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.077523, 0.098533, -0.246816, 0.072934),
            doubleArrayOf(-0.107647, 0.119149, -0.520935, 0.116003)
          )).equals(paramsErrors.getErrorsOf(layer.params.keys.weights)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the values weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.202771, -0.314063, -0.091634, -0.412156),
            doubleArrayOf(0.43209, -0.685103, -0.308845, -0.791595),
            doubleArrayOf(0.347967, -0.555616, -0.276653, -0.616254),
            doubleArrayOf(0.439844, -0.699312, -0.328048, -0.795263)
          )).equals(paramsErrors.getErrorsOf(layer.params.values.weights)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the queries biases") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.109727, 0.001192))
            .equals(paramsErrors.getErrorsOf(layer.params.queries.biases)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the keys biases") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0))
            .equals(paramsErrors.getErrorsOf(layer.params.keys.biases)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the values biases") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, -1.3, -1.0, -1.3))
            .equals(paramsErrors.getErrorsOf(layer.params.values.biases)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the first input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.057415, 0.586003, -0.788808, -0.138081))
            .equals(layer.inputArrays[0].errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the second input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.137857, 0.857493, -0.429106, -0.207021))
            .equals(layer.inputArrays[1].errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the third input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.127408, -0.047507, -0.313912, -0.052238))
            .equals(layer.inputArrays[2].errors, tolerance = 1.0e-06)
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
            doubleArrayOf(0.92, 1.1),
            doubleArrayOf(0.53, 1.04),
            doubleArrayOf(0.55, 1.03)
          )).equals(
            layer.queries.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected keys") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.96, 0.02),
            doubleArrayOf(0.18, -0.12),
            doubleArrayOf(-1.0, -0.56)
          )).equals(
            layer.keys.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.63, 0.5, -0.95, 0.32),
            doubleArrayOf(-0.2, 0.0, -0.13, -1.18),
            doubleArrayOf(-0.27, -0.2, -0.41, 1.4)
          )).equals(
            layer.values.values,
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected attention scores") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.582109, 0.314298, 0.103593),
            doubleArrayOf(0.503361, 0.339012, 0.157628),
            doubleArrayOf(0.506943, 0.338013, 0.155044)
          )).equals(
            DenseNDArrayFactory.fromRows(layer.attentionAct),
            tolerance = 1.0e-06)
        }
      }

      it("should match the expected output arrays") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.275899, 0.270336, -0.636335, -0.039567),
            doubleArrayOf(0.206755, 0.220155, -0.586891, -0.018279),
            doubleArrayOf(0.209909, 0.222462, -0.589105, -0.019572)
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
            doubleArrayOf(0.166976, 0.04868),
            doubleArrayOf(-0.157867, -0.044738),
            doubleArrayOf(-0.118836, -0.00275)
          )).equals(layer.queries.errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the keys") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.118337, -0.282144),
            doubleArrayOf(0.200448, 0.381435),
            doubleArrayOf(-0.082111, -0.099291)
          )).equals(layer.keys.errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.299378, -0.679784, -0.557768, -0.696967),
            doubleArrayOf(-0.254008, -0.432802, -0.321912, -0.42746),
            doubleArrayOf(-0.146614, -0.187414, -0.12032, -0.175574)
          )).equals(layer.values.errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the queries weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.023508, 0.065794, 0.260786, -0.049875),
            doubleArrayOf(0.002072, 0.004134, 0.075128, -0.007132)
          )).equals(paramsErrors.getErrorsOf(layer.params.queries.weights)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the keys weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.077523, 0.098533, -0.246816, 0.072934),
            doubleArrayOf(-0.107647, 0.119149, -0.520935, 0.116003)
          )).equals(paramsErrors.getErrorsOf(layer.params.keys.weights)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the values weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.202771, -0.314063, -0.091634, -0.412156),
            doubleArrayOf(0.43209, -0.685103, -0.308845, -0.791595),
            doubleArrayOf(0.347967, -0.555616, -0.276653, -0.616254),
            doubleArrayOf(0.439844, -0.699312, -0.328048, -0.795263)
          )).equals(paramsErrors.getErrorsOf(layer.params.values.weights)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the queries biases") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.109727, 0.001192))
            .equals(paramsErrors.getErrorsOf(layer.params.queries.biases)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the keys biases") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0))
            .equals(paramsErrors.getErrorsOf(layer.params.keys.biases)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the values biases") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, -1.3, -1.0, -1.3))
            .equals(paramsErrors.getErrorsOf(layer.params.values.biases)!!.values, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the first input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.057415, 0.586003, -0.788808, -0.138081))
            .equals(layer.inputArrays[0].errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the second input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.137857, 0.857493, -0.429106, -0.207021))
            .equals(layer.inputArrays[1].errors, tolerance = 1.0e-06)
        }
      }

      it("should match the expected errors of the third input") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.127408, -0.047507, -0.313912, -0.052238))
            .equals(layer.inputArrays[2].errors, tolerance = 1.0e-06)
        }
      }
    }
  }
})
