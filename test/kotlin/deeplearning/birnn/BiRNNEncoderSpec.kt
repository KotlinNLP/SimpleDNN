/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.birnn

import com.kotlinnlp.simplednn.core.layers.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import deeplearning.birnn.utils.BiRNNEncoderUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import kotlin.test.assertTrue

/**
 *
 */
class BiRNNEncoderSpec : Spek({

  describe("a BiRNNEncoder") {

    val inputSequence = BiRNNEncoderUtils.buildInputSequence()
    val birnn = BiRNNEncoderUtils.buildBiRNN()
    val encoder = BiRNNEncoder<DenseNDArray>(birnn)

    val encodedSequence = encoder.encode(inputSequence)

    it("should match the expected first output array") {
      assertTrue {
        encodedSequence[0].equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.187746, -0.50052, 0.109558, -0.005277, -0.084306, -0.628766)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected second output array") {
      assertTrue {
        encodedSequence[1].equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.704648, 0.200908, -0.064056, -0.329084, -0.237601, -0.449676)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected third output array") {
      assertTrue {
        encodedSequence[2].equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.256521, 0.725227, 0.781582, 0.129273, -0.716298, -0.263625)),
          tolerance = 1.0e-06
        )
      }
    }

    encoder.backward(
      outputErrorsSequence = BiRNNEncoderUtils.buildOutputErrorsSequence(),
      propagateToInput = true)

    val paramsErrors: BiRNNParameters = encoder.getParamsErrors()
    val l2rErrors = paramsErrors.leftToRight.paramsPerLayer[0] as SimpleRecurrentLayerParameters
    val r2lErrors = paramsErrors.rightToLeft.paramsPerLayer[0] as SimpleRecurrentLayerParameters

    it("should match the expected errors of the Left-to-right biases") {
      assertTrue {
        l2rErrors.unit.biases.values.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.071016, 0.268027, 0.345019)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Left-to-right weights") {
      assertTrue {
        (l2rErrors.unit.weights.values as DenseNDArray).equals(
          DenseNDArrayFactory.arrayOf(arrayOf(
            doubleArrayOf(0.001234, -0.107987),
            doubleArrayOf(0.175039, 0.015738),
            doubleArrayOf(0.213397, -0.046717)
          )),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Left-to-right recurrent weights") {
      assertTrue {
        l2rErrors.unit.recurrentWeights.values.equals(
          DenseNDArrayFactory.arrayOf(arrayOf(
            doubleArrayOf(0.041817, -0.059241, 0.013592),
            doubleArrayOf(0.042229, -0.086071, 0.019157),
            doubleArrayOf(0.035331, -0.11595, 0.02512)
          )),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Right-to-left biases") {
      assertTrue {
        r2lErrors.unit.biases.values.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.03906, 0.237598, -0.137858)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Right-to-left weights") {
      assertTrue {
        (r2lErrors.unit.weights.values as DenseNDArray).equals(
          DenseNDArrayFactory.arrayOf(arrayOf(
            doubleArrayOf(0.145713, 0.234548),
            doubleArrayOf(0.050135, 0.070768),
            doubleArrayOf(-0.06125, -0.017281)
          )),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Right-to-left recurrent weights") {
      assertTrue {
        r2lErrors.unit.recurrentWeights.values.equals(
          DenseNDArrayFactory.arrayOf(arrayOf(
            doubleArrayOf(-0.029278, -0.112568, -0.089725),
            doubleArrayOf(-0.074426, 0.003116, -0.070784),
            doubleArrayOf(0.022664, 0.040583, 0.044139)
          )),
          tolerance = 1.0e-06
        )
      }
    }
  }
})
