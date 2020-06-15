/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.birnn

import com.kotlinnlp.simplednn.core.layers.models.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import deeplearning.birnn.utils.BiRNNEncoderUtils
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class BiRNNEncoderSpec : Spek({

  describe("a BiRNNEncoder") {

    val inputSequence = BiRNNEncoderUtils.buildInputSequence()
    val birnn = BiRNNEncoderUtils.buildBiRNN()
    val encoder = BiRNNEncoder<DenseNDArray>(birnn, propagateToInput = true)

    val encodedSequence = encoder.forward(inputSequence)

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

    encoder.backward(outputErrors = BiRNNEncoderUtils.buildOutputErrorsSequence())

    val paramsErrors = encoder.getParamsErrors()

    val l2rParams = birnn.leftToRightNetwork.paramsPerLayer[0] as SimpleRecurrentLayerParameters
    val r2lParams = birnn.rightToLeftNetwork.paramsPerLayer[0] as SimpleRecurrentLayerParameters

    it("should match the expected errors of the Left-to-right biases") {
      assertTrue {
        paramsErrors.getErrorsOf(l2rParams.unit.biases)!!.values.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.213048, 0.804082, 1.035058)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Left-to-right weights") {
      assertTrue {
        (paramsErrors.getErrorsOf(l2rParams.unit.weights)!!.values as DenseNDArray).equals(
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.003701, -0.32396),
            doubleArrayOf(0.525116, 0.047213),
            doubleArrayOf(0.640192, -0.140151)
          )),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Left-to-right recurrent weights") {
      assertTrue {
        paramsErrors.getErrorsOf(l2rParams.unit.recurrentWeights)!!.values.equals(
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.125452, -0.177722, 0.040777),
            doubleArrayOf(0.126687, -0.258213, 0.057472),
            doubleArrayOf(0.105992, -0.347851, 0.07536)
          )),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Right-to-left biases") {
      assertTrue {
        paramsErrors.getErrorsOf(r2lParams.unit.biases)!!.values.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.117179, 0.712793, -0.413573)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Right-to-left weights") {
      assertTrue {
        (paramsErrors.getErrorsOf(r2lParams.unit.weights)!!.values as DenseNDArray).equals(
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(0.43714, 0.703645),
            doubleArrayOf(0.150406, 0.212304),
            doubleArrayOf(-0.18375, -0.051842)
          )),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the Right-to-left recurrent weights") {
      assertTrue {
        paramsErrors.getErrorsOf(r2lParams.unit.recurrentWeights)!!.values.equals(
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.087835, -0.337705, -0.269176),
            doubleArrayOf(-0.223279, 0.009348, -0.212353),
            doubleArrayOf(0.067992, 0.121748, 0.132418)
          )),
          tolerance = 1.0e-06
        )
      }
    }

    val inputErrors: List<DenseNDArray> = encoder.getInputErrors()

    it("should match the expected errors of first input array") {
      assertTrue {
        inputErrors[0].equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(1.031472, -0.627913)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of second input array") {
      assertTrue {
        inputErrors[1].equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.539497, -0.629167)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of third input array") {
      assertTrue {
        inputErrors[2].equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.013097, -0.09932)),
          tolerance = 1.0e-06
        )
      }
    }
  }
})
