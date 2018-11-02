/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.cfn

import com.kotlinnlp.simplednn.core.layers.models.recurrent.cfn.CFNLayerParameters
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class CFNLayerStructureSpec : Spek({

  describe("a CFNLayerStructure") {

    context("forward") {

      on("without previous state context") {

        val layer = CFNLayerStructureUtils.buildLayer(CFNLayerContextWindow.Empty())
        layer.forward()

        it("should match the expected input gate") {
          assertTrue {
            layer.inputGate.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.397, 0.252, 0.5, 0.705, 0.453)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected forget gate") {
          assertTrue {
            layer.forgetGate.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.853, 0.433, 0.116, 0.52, 0.242)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.675, -0.1, 0.762, 0.869, -0.804)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.268, -0.025, 0.381, 0.613, -0.364)),
              tolerance = 0.0005)
          }
        }
      }

      on("with previous state context") {

        val layer = CFNLayerStructureUtils.buildLayer(CFNLayerContextWindow.Back())
        layer.forward()

        it("should match the expected input gate") {
          assertTrue {
            layer.inputGate.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.723, 0.25, 0.55, 0.821, 0.535)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected forget gate") {
          assertTrue {
            layer.forgetGate.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.911, 0.181, 0.048, 0.675, 0.389)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected candidate") {
          assertTrue {
            layer.candidate.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.675, -0.1, 0.762, 0.869, -0.804)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.308, 0.011, 0.405, 0.230, -0.689)),
              tolerance = 0.0005)
          }
        }
      }
    }

    context("backward") {

      on("without previous and next state") {

        val layer = CFNLayerStructureUtils.buildLayer(CFNLayerContextWindow.Empty())
        val paramsErrors = CFNLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = CFNLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.302, -0.775, 0.531, -1.027, -0.814)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.049, 0.015, 0.101, -0.186, 0.162)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate") {
          assertTrue {
            layer.forgetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.065, -0.193, 0.111, -0.177, -0.13)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.inputGate.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.049, 0.015, 0.101, -0.186, 0.162)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate biases") {
          assertTrue {
            paramsErrors.forgetGate.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.039, 0.044, 0.044, -0.049),
                doubleArrayOf(-0.012, -0.013, -0.013, 0.015),
                doubleArrayOf(-0.081, -0.091, -0.091, 0.101),
                doubleArrayOf(0.149, 0.167, 0.167, -0.186),
                doubleArrayOf(-0.13, -0.146, -0.146, 0.162)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate weights") {
          assertTrue {
            (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.candidateWeights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.052, 0.059, 0.059, -0.065),
                doubleArrayOf(0.154, 0.174, 0.174, -0.193),
                doubleArrayOf(-0.089, -0.1, -0.1, 0.111),
                doubleArrayOf(0.142, 0.159, 0.159, -0.177),
                doubleArrayOf(0.104, 0.117, 0.117, -0.130)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertTrue {
            paramsErrors.inputGate.recurrentWeights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertTrue {
            paramsErrors.forgetGate.recurrentWeights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.318, 0.01, -0.027, 0.302)),
              tolerance = 0.0005)
          }
        }
      }

      on("with previous state only") {

        val layer = CFNLayerStructureUtils.buildLayer(CFNLayerContextWindow.Back())
        val paramsErrors = CFNLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = CFNLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.262, -0.739, 0.555, -1.41, -1.139)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.035, 0.014, 0.105, -0.18, 0.228)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate") {
          assertTrue {
            layer.forgetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.004, -0.022, -0.007, 0.222, 0.18)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.103, -0.183, 0.128, -0.283, -0.215)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.inputGate.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.035, 0.014, 0.105, -0.18, 0.228)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate biases") {
          assertTrue {
            paramsErrors.forgetGate.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.004, -0.022, -0.007, 0.222, 0.18)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.028, 0.032, 0.032, -0.035),
                doubleArrayOf(-0.011, -0.012, -0.012, 0.014),
                doubleArrayOf(-0.084, -0.094, -0.094, 0.105),
                doubleArrayOf(0.144, 0.162, 0.162, -0.18),
                doubleArrayOf(-0.182, -0.205, -0.205, 0.228)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate weights") {
          assertTrue {
            (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.003, -0.004, -0.004, 0.004),
                doubleArrayOf(0.017, 0.019, 0.019, -0.022),
                doubleArrayOf(0.006, 0.007, 0.007, -0.007),
                doubleArrayOf(-0.177, -0.199, -0.199, 0.222),
                doubleArrayOf(-0.144, -0.162, -0.162, 0.18)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.candidateWeights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.082, 0.093, 0.093, -0.103),
                doubleArrayOf(0.146, 0.164, 0.164, -0.183),
                doubleArrayOf(-0.102, -0.115, -0.115, 0.128),
                doubleArrayOf(0.226, 0.255, 0.255, -0.283),
                doubleArrayOf(0.172, 0.194, 0.194, -0.215)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertTrue {
            paramsErrors.inputGate.recurrentWeights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.007, -0.007, 0.011, 0.032, 0.028),
                doubleArrayOf(-0.003, 0.003, -0.004, -0.012, -0.011),
                doubleArrayOf(-0.021, 0.021, -0.031, -0.094, -0.084),
                doubleArrayOf(0.036, -0.036, 0.054, 0.162, 0.144),
                doubleArrayOf(-0.046, 0.046, -0.068, -0.205, -0.182)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertTrue {
            paramsErrors.forgetGate.recurrentWeights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.001, 0.001, -0.001, -0.004, -0.003),
                doubleArrayOf(0.004, -0.004, 0.006, 0.019, 0.017),
                doubleArrayOf(0.001, -0.001, 0.002, 0.007, 0.006),
                doubleArrayOf(-0.044, 0.044, -0.066, -0.199, -0.177),
                doubleArrayOf(-0.036, 0.036, -0.054, -0.162, -0.144)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.111, 0.37, -0.281, 0.126)),
              tolerance = 0.0005)
          }
        }
      }

      on("with next state only") {

        val layer = CFNLayerStructureUtils.buildLayer(CFNLayerContextWindow.Front(
          currentLayerOutput = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.261, -0.025, 0.363, 0.546, -0.349))))
        val paramsErrors = CFNLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = CFNLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.451, -0.425, 0.97, -1.710, -1.016)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.073, 0.008, 0.185, -0.309, 0.202)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate") {
          assertTrue {
            layer.forgetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.097, -0.106, 0.204, -0.295, -0.163)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.inputGate.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.073, 0.008, 0.185, -0.309, 0.202)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate biases") {
          assertTrue {
            paramsErrors.forgetGate.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.058, 0.066, 0.066, -0.073),
                doubleArrayOf(-0.006, -0.007, -0.007, 0.008),
                doubleArrayOf(-0.148, -0.166, -0.166, 0.185),
                doubleArrayOf(0.2479, 0.278, 0.278, -0.309),
                doubleArrayOf(-0.162, -0.182, -0.182, 0.202)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate weights") {
          assertTrue {
            (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.candidateWeights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.078, 0.088, 0.088, -0.097),
                doubleArrayOf(0.085, 0.095, 0.095, -0.106),
                doubleArrayOf(-0.163, -0.183, -0.183, 0.204),
                doubleArrayOf(0.236, 0.265, 0.265, -0.295),
                doubleArrayOf(0.130, 0.146, 0.146, -0.163)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertTrue {
            paramsErrors.inputGate.recurrentWeights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertTrue {
            paramsErrors.forgetGate.recurrentWeights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.378, 0.135, -0.114, 0.372)),
              tolerance = 0.0005)
          }
        }
      }

      on("with previous and next state") {

        val layer = CFNLayerStructureUtils.buildLayer(CFNLayerContextWindow.Bilateral(
          currentLayerOutput = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.299, 0.0108, 0.384, 0.226, -0.597))))
        val paramsErrors = CFNLayerParameters(inputSize = 4, outputSize = 5)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = CFNLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        layer.backward(paramsErrors = paramsErrors, propagateToInput = true)

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.41, -0.389, 0.999, -2.232, -1.364)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate") {
          assertTrue {
            layer.inputGate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.055, 0.007, 0.188, -0.286, 0.273)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate") {
          assertTrue {
            layer.forgetGate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.007, -0.011, -0.013, 0.351, 0.215)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the candidate") {
          assertTrue {
            layer.candidate.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.161, -0.096, 0.231, -0.448, -0.258)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate biases") {
          assertTrue {
            paramsErrors.inputGate.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.055, 0.007, 0.188, -0.286, 0.273)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate biases") {
          assertTrue {
            paramsErrors.forgetGate.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.007, -0.011, -0.013, 0.351, 0.215)),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate weights") {
          assertTrue {
            (paramsErrors.inputGate.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.044, 0.05, 0.05, -0.055),
                doubleArrayOf(-0.006, -0.007, -0.007, 0.007),
                doubleArrayOf(-0.151, -0.169, -0.169, 0.188),
                doubleArrayOf(0.229, 0.257, 0.257, -0.286),
                doubleArrayOf(-0.218, -0.246, -0.246, 0.273)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate weights") {
          assertTrue {
            (paramsErrors.forgetGate.weights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.005, -0.006, -0.006, 0.007),
                doubleArrayOf(0.009, 0.01, 0.01, -0.011),
                doubleArrayOf(0.011, 0.012, 0.012, -0.013),
                doubleArrayOf(-0.281, -0.316, -0.316, 0.351),
                doubleArrayOf(-0.172, -0.194, -0.194, 0.215)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the candidate weights") {
          assertTrue {
            (paramsErrors.candidateWeights.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.129, 0.145, 0.145, -0.161),
                doubleArrayOf(0.077, 0.087, 0.087, -0.096),
                doubleArrayOf(-0.185, -0.208, -0.208, 0.231),
                doubleArrayOf(0.358, 0.403, 0.403, -0.448),
                doubleArrayOf(0.206, 0.232, 0.232, -0.258)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the input gate recurrent weights") {
          assertTrue {
            paramsErrors.inputGate.recurrentWeights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.011, -0.011, 0.017, 0.05, 0.044),
                doubleArrayOf(-0.001, 0.001, -0.002, -0.007, -0.006),
                doubleArrayOf(-0.038, 0.038, -0.056, -0.169, -0.151),
                doubleArrayOf(0.057, -0.057, 0.086, 0.257, 0.229),
                doubleArrayOf(-0.055, 0.055, -0.082, -0.246, -0.218)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the forget gate recurrent weights") {
          assertTrue {
            paramsErrors.forgetGate.recurrentWeights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.001, 0.001, -0.002, -0.006, -0.005),
                doubleArrayOf(0.002, -0.002, 0.003, 0.01, 0.009),
                doubleArrayOf(0.003, -0.003, 0.004, 0.012, 0.011),
                doubleArrayOf(-0.07, 0.07, -0.105, -0.316, -0.281),
                doubleArrayOf(-0.043, 0.043, -0.065, -0.194, -0.172)
              )),
              tolerance = 0.0005)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.123, 0.625, -0.467, 0.104)),
              tolerance = 0.0005)
          }
        }
      }
    }
  }
})
