/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.birnn

import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
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
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.706651, 0.283582)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected second output array") {
      assertTrue {
        encodedSequence[1].equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.640841, 0.159521)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected third output array") {
      assertTrue {
        encodedSequence[2].equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.684963, 0.537581)),
          tolerance = 1.0e-06
        )
      }
    }

    encoder.backward(
      outputErrorsSequence = BiRNNEncoderUtils.buildOutputErrorsSequence(),
      propagateToInput = true)

    val paramsErrors: BiRNNParameters = encoder.getParamsErrors()
    val feedforwardErrors = paramsErrors.output.paramsPerLayer[0] as FeedforwardLayerParameters
    val l2rErrors = paramsErrors.leftToRight.paramsPerLayer[0] as SimpleRecurrentLayerParameters
    val r2lErrors = paramsErrors.rightToLeft.paramsPerLayer[0] as SimpleRecurrentLayerParameters

    it("should match the expected errors of the output biases") {
      assertTrue {
        feedforwardErrors.unit.biases.values.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.034108, -0.088121)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the output weights") {
      assertTrue {
        (feedforwardErrors.unit.weights.values as DenseNDArray).equals(
          DenseNDArrayFactory.arrayOf(arrayOf(
            doubleArrayOf(0.039544, -0.055866, -0.013748, 0.013699, 0.029292, 0.001322),
            doubleArrayOf(-0.021673, -0.047306, -0.059771, -0.009569, 0.054561, 0.028176)
          )),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Left-to-right biases") {
      assertTrue {
        l2rErrors.unit.biases.values.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.020768, -0.119312, -0.053248)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Left-to-right weights") {
      assertTrue {
        (l2rErrors.unit.weights.values as DenseNDArray).equals(
          DenseNDArrayFactory.arrayOf(arrayOf(
            doubleArrayOf(0.011778, 0.066219),
            doubleArrayOf(-0.07297, 0.000163),
            doubleArrayOf(-0.016202, 0.007631)
          )),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Left-to-right recurrent weights") {
      assertTrue {
        l2rErrors.unit.recurrentWeights.values.equals(
          DenseNDArrayFactory.arrayOf(arrayOf(
            doubleArrayOf(0.032536, -0.003084, 0.001676),
            doubleArrayOf(-0.012177, 0.034751, -0.007579),
            doubleArrayOf(0.015461, -0.000738, 0.000646)
          )),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Right-to-left biases") {
      assertTrue {
        r2lErrors.unit.biases.values.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.063832, 0.017479, -0.002648)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Right-to-left weights") {
      assertTrue {
        (r2lErrors.unit.weights.values as DenseNDArray).equals(
          DenseNDArrayFactory.arrayOf(arrayOf(
            doubleArrayOf(-0.006257, -0.006522),
            doubleArrayOf(-0.011474, 0.00725),
            doubleArrayOf(-0.023557, -0.043552)
          )),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Right-to-left recurrent weights") {
      assertTrue {
        r2lErrors.unit.recurrentWeights.values.equals(
          DenseNDArrayFactory.arrayOf(arrayOf(
            doubleArrayOf(0.017679, -0.005845, 0.014038),
            doubleArrayOf(-0.011591, 0.017901, -0.001554),
            doubleArrayOf(0.009355, 0.015193, 0.017372)
          )),
          tolerance = 1.0e-06
        )
      }
    }
  }
})
