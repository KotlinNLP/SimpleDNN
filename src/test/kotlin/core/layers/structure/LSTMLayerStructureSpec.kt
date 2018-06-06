/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.structure

import com.kotlinnlp.simplednn.core.layers.recurrent.lstm.LSTMLayerParameters
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import core.layers.structure.utils.LSTMLayerStructureUtils
import core.layers.structure.contextwindows.LSTMLayerContextWindow
import kotlin.test.assertEquals

/**
 *
 */
class LSTMLayerStructureSpec : Spek({

  describe("a LSTMLayerStructure") {

    context("forward") {

      on("without previous state context") {

        val layer = LSTMLayerStructureUtils.buildLayer(LSTMLayerContextWindow.Empty())
        layer.forward()

        it("should match the expected input gate") {
          assertEquals(true, layer.inputGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.40, 0.25, 0.50, 0.70, 0.45)),
            tolerance = 0.005))
        }

        it("should match the expected output gate") {
          assertEquals(true, layer.outputGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.85, 0.43, 0.12, 0.52, 0.24)),
            tolerance = 0.005))
        }

        it("should match the expected forget gate") {
          assertEquals(true, layer.forgetGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.79, 0.35, 0.88, 0.85, 0.45)),
            tolerance = 0.005))
        }

        it("should match the expected candidate") {
          assertEquals(true, layer.candidate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.38, -0.45, -0.92, 0.98, -0.89)),
            tolerance = 0.005))
        }

        it("should match the expected cell") {
          assertEquals(true, layer.cell.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.15, -0.11, -0.43, 0.6, -0.38)),
            tolerance = 0.005))
        }

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.13, -0.05, -0.05, 0.31, -0.09)),
            tolerance = 0.005))
        }
      }

      on("with previous state context") {

        val layer = LSTMLayerStructureUtils.buildLayer(LSTMLayerContextWindow.Back())
        layer.forward()

        it("should match the expected input gate") {
          assertEquals(true, layer.inputGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.72, 0.25, 0.55, 0.82, 0.53)),
            tolerance = 0.005))
        }

        it("should match the expected output gate") {
          assertEquals(true, layer.outputGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.91, 0.18, 0.05, 0.67, 0.39)),
            tolerance = 0.005))
        }

        it("should match the expected forget gate") {
          assertEquals(true, layer.forgetGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.91, 0.62, 0.84, 0.91, 0.62)),
            tolerance = 0.005))
        }

        it("should match the expected candidate") {
          assertEquals(true, layer.candidate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.23, 0.33, -0.95, 0.99, -0.93)),
            tolerance = 0.005))
        }

        it("should match the expected cell") {
          assertEquals(true, layer.cell.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.51, -0.28, 0.31, 0.72, -0.41)),
            tolerance = 0.005))
        }

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.47, -0.05, 0.01, 0.48, -0.16)),
            tolerance = 0.005))
        }
      }

      on("with init hidden layer") {

        val contextWindow = LSTMLayerContextWindow.BackHidden()
        val layer = LSTMLayerStructureUtils.buildLayer(contextWindow)

        contextWindow.setRefLayer(layer)
        
        layer.forward()

        it("should match the expected input gate") {
          assertEquals(true, layer.inputGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.72, 0.25, 0.55, 0.82, 0.53)),
            tolerance = 0.005))
        }

        it("should match the expected output gate") {
          assertEquals(true, layer.outputGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.91, 0.18, 0.05, 0.67, 0.39)),
            tolerance = 0.005))
        }

        it("should match the expected forget gate") {
          assertEquals(true, layer.forgetGate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.91, 0.62, 0.84, 0.91, 0.62)),
            tolerance = 0.005))
        }

        it("should match the expected candidate") {
          assertEquals(true, layer.candidate.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.23, 0.33, -0.95, 0.99, -0.93)),
            tolerance = 0.005))
        }

        it("should match the expected cell") {
          assertEquals(true, layer.cell.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.16, 0.08, -0.48, 0.67, -0.46)),
            tolerance = 0.005))
        }

        it("should match the expected outputArray") {
          assertEquals(true, layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.15, 0.01, -0.02, 0.45, -0.18)),
            tolerance = 0.005))
        }
      }
    }

    context("backward") {

      on("without previous and next state") {

        val layer = LSTMLayerStructureUtils.buildLayer(LSTMLayerContextWindow.Empty())
        val paramsErrors = LSTMLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LSTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.7, -0.8, 0.1, -1.33, -0.54)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the cell") {
          assertEquals(true, layer.cell.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.58, -0.34, 0.01, -0.44, -0.11)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.20, -0.07, 0.0, -0.01, -0.01)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.05, 0.03, 0.0, -0.09, 0.02)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the output gate") {
          assertEquals(true, layer.outputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.01, 0.02, 0.0, -0.2, 0.04)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.20, -0.07, 0.0, -0.01, -0.01)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.05, 0.03, 0.0, -0.09, 0.02)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the output gate biases") {
          assertEquals(true, paramsErrors.outputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.01, 0.02, 0.0, -0.2, 0.04)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.16, 0.18, 0.18, -0.2),
              doubleArrayOf(0.05, 0.06, 0.06, -0.07),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.01, 0.01, 0.01, -0.01),
              doubleArrayOf(0.01, 0.01, 0.01, -0.01)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.04, -0.05, -0.05, 0.05),
              doubleArrayOf(-0.02, -0.03, -0.03, 0.03),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.07, 0.08, 0.08, -0.09),
              doubleArrayOf(-0.02, -0.02, -0.02, 0.02)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the output gate weights") {
          assertEquals(true, (paramsErrors.outputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.01, -0.01, -0.01, 0.01),
              doubleArrayOf(-0.02, -0.02, -0.02, 0.02),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.16, 0.18, 0.18, -0.2),
              doubleArrayOf(-0.03, -0.03, -0.03, 0.04)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the candidate recurrent weights") {
          assertEquals(true, paramsErrors.candidate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the output gate recurrent weights") {
          assertEquals(true, paramsErrors.outputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.12, -0.14, 0.03, 0.02)),
            tolerance = 0.005))
        }
      }

      on("with previous state only") {

        val layer = LSTMLayerStructureUtils.buildLayer(LSTMLayerContextWindow.Back())
        val paramsErrors = LSTMLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LSTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.104, -0.801, 0.165, -1.156, -0.609)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the cell") {
          assertEquals(true, layer.cell.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.07, -0.133, 0.007, -0.378, -0.198)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.048, -0.03, 0.0, -0.006, -0.015)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.003, -0.008, -0.002, -0.055, 0.046)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the output gate") {
          assertEquals(true, layer.outputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.004, 0.033, 0.002, -0.182, 0.059)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.005, 0.019, 0.001, -0.003, -0.005)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.048, -0.03, 0.0, -0.006, -0.015)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.003, -0.008, -0.002, -0.055, 0.046)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the output gate biases") {
          assertEquals(true, paramsErrors.outputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.004, 0.033, 0.002, -0.182, 0.059)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.005, 0.019, 0.001, -0.003, -0.005)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.038, 0.043, 0.043, -0.048),
              doubleArrayOf(0.024, 0.027, 0.027, -0.03),
              doubleArrayOf(0.00, 0.00, 0.00, 0.00),
              doubleArrayOf(0.005, 0.006, 0.006, -0.006),
              doubleArrayOf(0.012, 0.013, 0.013, -0.015)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.003, -0.003, -0.003, 0.003),
              doubleArrayOf(0.007, 0.007, 0.007, -0.008),
              doubleArrayOf(0.001, 0.002, 0.002, -0.002),
              doubleArrayOf(0.044, 0.05, 0.05, -0.055),
              doubleArrayOf(-0.036, -0.041, -0.041, 0.046)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the output gate weights") {
          assertEquals(true, (paramsErrors.outputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.003, 0.004, 0.004, -0.004),
              doubleArrayOf(-0.027, -0.03, -0.03, 0.033),
              doubleArrayOf(-0.002, -0.002, -0.002, 0.002),
              doubleArrayOf(0.146, 0.164, 0.164, -0.182),
              doubleArrayOf(-0.047, -0.053, -0.053, 0.059)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.004, 0.004, 0.004, -0.005),
              doubleArrayOf(-0.015, -0.017, -0.017, 0.019),
              doubleArrayOf(-0.001, -0.001, -0.001, 0.001),
              doubleArrayOf(0.003, 0.003, 0.003, -0.003),
              doubleArrayOf(0.004, 0.004, 0.004, -0.005)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate recurrent weights") {
          assertEquals(true, paramsErrors.candidate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.01, -0.01, 0.014, 0.043, 0.038),
              doubleArrayOf(0.006, -0.006, 0.009, 0.027, 0.024),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.001, -0.001, 0.002, 0.006, 0.005),
              doubleArrayOf(0.003, -0.003, 0.004, 0.013, 0.012)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.001, 0.001, -0.001, -0.003, -0.003),
              doubleArrayOf(0.002, -0.002, 0.002, 0.007, 0.007),
              doubleArrayOf(0.0, 0.0, 0.001, 0.002, 0.001),
              doubleArrayOf(0.011, -0.011, 0.017, 0.05, 0.044),
              doubleArrayOf(-0.009, 0.009, -0.014, -0.041, -0.036)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the output gate recurrent weights") {
          assertEquals(true, paramsErrors.outputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.001, -0.001, 0.001, 0.004, 0.003),
              doubleArrayOf(-0.007, 0.007, -0.01, -0.03, -0.027),
              doubleArrayOf(0.0, 0.0, -0.001, -0.002, -0.002),
              doubleArrayOf(0.036, -0.036, 0.055, 0.164, 0.146),
              doubleArrayOf(-0.012, 0.012, -0.018, -0.053, -0.047)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.001, -0.001, 0.001, 0.004, 0.004),
              doubleArrayOf(-0.004, 0.004, -0.006, -0.017, -0.015),
              doubleArrayOf(0.0, 0.0, 0.0, -0.001, -0.001),
              doubleArrayOf(0.001, -0.001, 0.001, 0.003, 0.003),
              doubleArrayOf(0.001, -0.001, 0.001, 0.004, 0.004)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.106, -0.055, 0.002, 0.058)),
            tolerance = 0.0005))
        }
      }

      on("with init hidden") {

        val contextWindow = LSTMLayerContextWindow.BackHidden()
        val layer = LSTMLayerStructureUtils.buildLayer(contextWindow)
        val paramsErrors = LSTMLayerParameters(inputSize = 4, outputSize = 5)

        contextWindow.setRefLayer(layer)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LSTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.718, -0.735, 0.127, -1.187, -0.629)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.002, -0.253, -0.036, 0.006)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the init hidden array") {
          assertEquals(true, contextWindow.getPrevStateLayer().getInitHiddenErrors().equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.020, -0.051, 0.017, -0.264, 0.449)),
            tolerance = 0.0005))
        }
      }

      on("with next state only") {

        val layer = LSTMLayerStructureUtils.buildLayer(LSTMLayerContextWindow.Front())
        val paramsErrors = LSTMLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LSTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.188, -0.849, 1.42, -1.998, -2.242)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the cell") {
          assertEquals(true, layer.cell.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.066, -0.683, 1.034, -0.346, -0.704)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.023, -0.136, 0.081, -0.009, -0.068)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.006, 0.058, -0.238, -0.071, 0.155)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the output gate") {
          assertEquals(true, layer.outputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.004, 0.024, -0.063, -0.299, 0.157)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.023, -0.136, 0.081, -0.009, -0.068)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.006, 0.058, -0.238, -0.071, 0.155)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the output gate biases") {
          assertEquals(true, paramsErrors.outputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.004, 0.024, -0.063, -0.299, 0.157)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.018, 0.02, 0.02, -0.023),
              doubleArrayOf(0.109, 0.123, 0.123, -0.136),
              doubleArrayOf(-0.065, -0.073, -0.073, 0.081),
              doubleArrayOf(0.007, 0.008, 0.008, -0.009),
              doubleArrayOf(0.054, 0.061, 0.061, -0.068)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.005, -0.005, -0.005, 0.006),
              doubleArrayOf(-0.047, -0.053, -0.053, 0.058),
              doubleArrayOf(0.19, 0.214, 0.214, -0.238),
              doubleArrayOf(0.057, 0.064, 0.064, -0.071),
              doubleArrayOf(-0.124, -0.139, -0.139, 0.155)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the output gate weights") {
          assertEquals(true, (paramsErrors.outputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.003, -0.003, -0.003, 0.004),
              doubleArrayOf(-0.019, -0.021, -0.021, 0.024),
              doubleArrayOf(0.05, 0.056, 0.056, -0.063),
              doubleArrayOf(0.239, 0.269, 0.269, -0.299),
              doubleArrayOf(-0.126, -0.141, -0.141, 0.157)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate recurrent weights") {
          assertEquals(true, paramsErrors.candidate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the output gate recurrent weights") {
          assertEquals(true, paramsErrors.outputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
              doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.023, 0.106, -0.06, -0.003)),
            tolerance = 0.0005))
        }
      }

      on("with previous and next state") {

        val layer = LSTMLayerStructureUtils.buildLayer(LSTMLayerContextWindow.Bilateral())
        val paramsErrors = LSTMLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LSTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

        it("should match the expected errors of the outputArray") {
          assertEquals(true, layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.406, -0.851, 1.485, -1.826, -2.309)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the cell") {
          assertEquals(true, layer.cell.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.363, -0.462, 0.965, -0.277, -0.989)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate") {
          assertEquals(true, layer.candidate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.249, -0.103, 0.053, -0.004, -0.074)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate") {
          assertEquals(true, layer.inputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.016, -0.028, -0.227, -0.04, 0.228)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the output gate") {
          assertEquals(true, layer.outputGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.017, 0.035, 0.021, -0.288, 0.225)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate") {
          assertEquals(true, layer.forgetGate.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.024, 0.065, 0.13, -0.002, -0.023)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate biases") {
          assertEquals(true, paramsErrors.candidate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.249, -0.103, 0.053, -0.004, -0.074)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate biases") {
          assertEquals(true, paramsErrors.inputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.016, -0.028, -0.227, -0.04, 0.228)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the output gate biases") {
          assertEquals(true, paramsErrors.outputGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.017, 0.035, 0.021, -0.288, 0.225)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate biases") {
          assertEquals(true, paramsErrors.forgetGate.biases.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.024, 0.065, 0.13, -0.002, -0.023)),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate weights") {
          assertEquals(true, (paramsErrors.candidate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.199, -0.224, -0.224, 0.249),
              doubleArrayOf(0.082, 0.093, 0.093, -0.103),
              doubleArrayOf(-0.042, -0.048, -0.048, 0.053),
              doubleArrayOf(0.004, 0.004, 0.004, -0.004),
              doubleArrayOf(0.059, 0.067, 0.067, -0.074)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate weights") {
          assertEquals(true, (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.013, 0.015, 0.015, -0.016),
              doubleArrayOf(0.023, 0.026, 0.026, -0.028),
              doubleArrayOf(0.181, 0.204, 0.204, -0.227),
              doubleArrayOf(0.032, 0.036, 0.036, -0.04),
              doubleArrayOf(-0.182, -0.205, -0.205, 0.228)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the output gate weights") {
          assertEquals(true, (paramsErrors.outputGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.013, -0.015, -0.015, 0.017),
              doubleArrayOf(-0.028, -0.032, -0.032, 0.035),
              doubleArrayOf(-0.017, -0.019, -0.019, 0.021),
              doubleArrayOf(0.23, 0.259, 0.259, -0.288),
              doubleArrayOf(-0.18, -0.202, -0.202, 0.225)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate weights") {
          assertEquals(true, (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.019, -0.021, -0.021, 0.024),
              doubleArrayOf(-0.052, -0.059, -0.059, 0.065),
              doubleArrayOf(-0.104, -0.117, -0.117, 0.13),
              doubleArrayOf(0.002, 0.002, 0.002, -0.002),
              doubleArrayOf(0.019, 0.021, 0.021, -0.023)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the candidate recurrent weights") {
          assertEquals(true, paramsErrors.candidate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.05, 0.05, -0.075, -0.224, -0.199),
              doubleArrayOf(0.021, -0.021, 0.031, 0.093, 0.082),
              doubleArrayOf(-0.011, 0.011, -0.016, -0.048, -0.042),
              doubleArrayOf(0.001, -0.001, 0.001, 0.004, 0.004),
              doubleArrayOf(0.015, -0.015, 0.022, 0.067, 0.059)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertEquals(true, paramsErrors.inputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(0.003, -0.003, 0.005, 0.015, 0.013),
              doubleArrayOf(0.006, -0.006, 0.009, 0.026, 0.023),
              doubleArrayOf(0.045, -0.045, 0.068, 0.204, 0.181),
              doubleArrayOf(0.008, -0.008, 0.012, 0.036, 0.032),
              doubleArrayOf(-0.046, 0.046, -0.068, -0.205, -0.182)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the output gate recurrent weights") {
          assertEquals(true, paramsErrors.outputGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.003, 0.003, -0.005, -0.015, -0.013),
              doubleArrayOf(-0.007, 0.007, -0.011, -0.032, -0.028),
              doubleArrayOf(-0.004, 0.004, -0.006, -0.019, -0.017),
              doubleArrayOf(0.058, -0.058, 0.086, 0.259, 0.23),
              doubleArrayOf(-0.045, 0.045, -0.067, -0.202, -0.18)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertEquals(true, paramsErrors.forgetGate.recurrentWeights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              doubleArrayOf(-0.005, 0.005, -0.007, -0.021, -0.019),
              doubleArrayOf(-0.013, 0.013, -0.02, -0.059, -0.052),
              doubleArrayOf(-0.026, 0.026, -0.039, -0.117, -0.104),
              doubleArrayOf(0.0, 0.0, 0.001, 0.002, 0.002),
              doubleArrayOf(0.005, -0.005, 0.007, 0.021, 0.019)
            )),
            tolerance = 0.0005))
        }

        it("should match the expected errors of the inputArray") {
          assertEquals(true, layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(0.042, 0.388, -0.243, 0.181)),
            tolerance = 0.0005))
        }
      }
    }
  }
})
