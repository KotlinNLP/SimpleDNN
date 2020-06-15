/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.neuralprocessor.batchfeedforward

import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue


/**
 *
 */
class BatchFeedforwardProcessorSpec : Spek({

  describe("a BatchFeedforwardProcessor") {

    val inputSequence = BatchFeedforwardUtils.buildInputBatch()
    val model = BatchFeedforwardUtils.buildParams()
    val processor = BatchFeedforwardProcessor<DenseNDArray>(model = model, propagateToInput = true)
    val output = processor.forward(inputSequence)

    it("should match the expected first output array") {
      assertTrue {
        output[0].equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.66959, -0.793199)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected second output array") {
      assertTrue {
        output[1].equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.739783, 0.197375)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected third output array") {
      assertTrue {
        output[2].equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.318521, -0.591519)),
          tolerance = 1.0e-06
        )
      }
    }

    processor.backward(outputErrors = BatchFeedforwardUtils.buildOutputErrors())

    val paramsErrors = processor.getParamsErrors()

    val params = model.paramsPerLayer[0] as FeedforwardLayerParameters

    it("should match the expected errors of the biases") {
      assertTrue {
        paramsErrors.getErrorsOf(params.unit.biases)!!.values.equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.290168, -0.659261)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of the weights") {
      assertTrue {
        (paramsErrors.getErrorsOf(params.unit.weights)!!.values).equals(
          DenseNDArrayFactory.arrayOf(listOf(
            doubleArrayOf(-0.259834, 0.293235, -0.283416),
            doubleArrayOf(-0.497742, -0.197778, 0.41039)
          )),
          tolerance = 1.0e-06
        )
      }
    }

    val inputErrors: List<DenseNDArray> = processor.getInputErrors()

    it("should match the expected errors of first input array") {
      assertTrue {
        inputErrors[0].equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.329642, 0.160346, -0.415821)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of second input array") {
      assertTrue {
        inputErrors[1].equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(0.221833, -0.095071, 0.316905)),
          tolerance = 1.0e-06
        )
      }
    }

    it("should match the expected errors of third input array") {
      assertTrue {
        inputErrors[2].equals(
          DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.216483, 0.243231, 0.12538)),
          tolerance = 1.0e-06
        )
      }
    }
  }
})
