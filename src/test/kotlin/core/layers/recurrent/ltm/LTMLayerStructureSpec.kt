/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.ltm

import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class LTMLayerStructureSpec : Spek({

  describe("a LTMLayer") {

    context("forward") {

      context("without previous state context") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayersWindow.Empty)
        layer.forward()

        it("should match the expected input gate L1") {
          assertTrue {
            layer.inputGate1.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.305764, 0.251618, 0.574443, 0.517493)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected input gate L2") {
          assertTrue {
            layer.inputGate2.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.702661, 0.384616, 0.244161, 0.470036)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected input gate L3") {
          assertTrue {
            layer.inputGate3.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.694236, 0.475021, 0.731059, 0.790841)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected cell") {
          assertTrue {
            layer.cell.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.531299, 0.439948, 0.484336, 0.44371)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.368847, 0.208984, 0.354078, 0.350904)),
              tolerance = 1.0e-06)
          }
        }
      }

      context("with previous state context") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayersWindow.Back)
        layer.forward()

        it("should match the expected input gate L1") {
          assertTrue {
            layer.inputGate1.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.495, 0.349781, 0.372852, 0.455121)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected input gate L2") {
          assertTrue {
            layer.inputGate2.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.702661, 0.336261, 0.334033, 0.645656)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected input gate L3") {
          assertTrue {
            layer.inputGate3.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.706822, 0.631812, 0.547358, 0.603483)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected cell") {
          assertTrue {
            layer.cell.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.639367, 0.243846, 0.477747, 0.209972)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.451919, 0.154065, 0.261499, 0.126715)),
              tolerance = 1.0e-06)
          }
        }
      }
    }

    context("backward") {

      context("without previous and next state") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayersWindow.Empty)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.201153, -0.541016, 0.504078, -1.289096)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.034775, -0.063322, 0.092037, -0.251637)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.178819, 0.073623, 0.263618, -0.084123)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.026672, 0.005332, 0.015735, -0.009873)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.011423, 0.004385, 0.027947, -0.010844)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.022686, -0.059356, 0.048001, -0.094613)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.021337, -0.024005, -0.024005, 0.026672),
                doubleArrayOf(-0.004266, -0.004799, -0.004799, 0.005332),
                doubleArrayOf(-0.012588, -0.014161, -0.014161, 0.015735),
                doubleArrayOf(0.007898, 0.008886, 0.008886, -0.009873)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.009139, -0.010281, -0.010281, 0.011423),
                doubleArrayOf(-0.003508, -0.003946, -0.003946, 0.004385),
                doubleArrayOf(-0.022357, -0.025152, -0.025152, 0.027947),
                doubleArrayOf(0.008675, 0.009760, 0.009760, -0.010844)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.018149, 0.020417, 0.020417, -0.022686),
                doubleArrayOf(0.047485, 0.053421, 0.053421, -0.059356),
                doubleArrayOf(-0.038401, -0.043201, -0.043201, 0.048001),
                doubleArrayOf(0.075690, 0.085152, 0.085152, -0.094613)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.007471, -0.003365, -0.004877, -0.008459),
                doubleArrayOf(-0.013605, -0.006128, -0.008881, -0.015402),
                doubleArrayOf(0.019774, 0.008907, 0.012909, 0.022387),
                doubleArrayOf(-0.054064, -0.024353, -0.035294, -0.061208)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.226967, 0.009912, -0.105134, -0.040795)),
              tolerance = 1.0e-06)
          }
        }
      }

      context("with previous state only") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayersWindow.Back)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.118081, -0.595935, 0.411499, -1.513285)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.019244, -0.069425, 0.056198, -0.151492)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.104782, 0.054728, 0.184063, -0.035139)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.018405, 0.004185, 0.014377, -0.005626)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.010837, 0.004273, 0.015267, -0.003659)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.015645, -0.033804, 0.048707, -0.076034)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.018405, -0.012883, -0.022086, 0.00184),
                doubleArrayOf(-0.004185, -0.00293, -0.005023, 0.000419),
                doubleArrayOf(-0.014377, -0.010064, -0.017252, 0.001438),
                doubleArrayOf(0.005626, 0.003938, 0.006751, -0.000563)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.010837, -0.007586, -0.013004, 0.001084),
                doubleArrayOf(-0.004273, -0.002991, -0.005127, 0.000427),
                doubleArrayOf(-0.015267, -0.010687, -0.018320, 0.001527),
                doubleArrayOf(0.003659, 0.002561, 0.004391, -0.000366)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.015645, 0.010951, 0.018774, -0.001564),
                doubleArrayOf(0.033804, 0.023663, 0.040565, -0.003380),
                doubleArrayOf(-0.048707, -0.034095, -0.058449, 0.004871),
                doubleArrayOf(0.076034, 0.053224, 0.091241, -0.007603)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.019972, -0.009083, -0.016466, -0.015758),
                doubleArrayOf(-0.072048, -0.032766, -0.059400, -0.056847),
                doubleArrayOf(0.058321, 0.026523, 0.048083, 0.046016),
                doubleArrayOf(-0.157217, -0.071498, -0.129617, -0.124046)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.165703, 0.006373, -0.085227, -0.025508)),
              tolerance = 1.0e-06)
          }
        }
      }

      context("with next state only") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayersWindow.Front())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.498847, -0.841016, 0.304078, -0.989096)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.08624, 0.12332, 0.105471, -0.316492)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.272226, 0.109696, 0.165076, -0.190172)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.040604, 0.007945, 0.009853, -0.02232)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.017391, 0.006533, 0.0175, -0.024515)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.05626, -0.09227, 0.028956, -0.072595)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.032483, -0.036544, -0.036544, 0.040604),
                doubleArrayOf(-0.006356, -0.00715, -0.00715, 0.007945),
                doubleArrayOf(-0.007882, -0.008868, -0.008868, 0.009853),
                doubleArrayOf(0.017856, 0.020088, 0.020088, -0.02232)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.013912, -0.015652, -0.015652, 0.017391),
                doubleArrayOf(-0.005226, -0.00588, -0.005880, 0.006533),
                doubleArrayOf(-0.014, -0.01575, -0.01575, 0.0175),
                doubleArrayOf(0.019612, 0.022063, 0.022063, -0.024515)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.045008, -0.050634, -0.050634, 0.05626),
                doubleArrayOf(0.073816, 0.083043, 0.083043, -0.09227),
                doubleArrayOf(-0.023165, -0.026061, -0.026061, 0.028956),
                doubleArrayOf(0.058076, 0.065335, 0.065335, -0.072595)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.018529, 0.008346, 0.012096, 0.020977),
                doubleArrayOf(0.026495, 0.011934, 0.017296, 0.029996),
                doubleArrayOf(0.02266, 0.010207, 0.014793, 0.025655),
                doubleArrayOf(-0.067998, -0.030629, -0.044390, -0.076983)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.14514, 0.004807, -0.08452, -0.013372)),
              tolerance = 1.0e-06)
          }
        }
      }

      context("with previous and next state") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayersWindow.Bilateral)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.581919, -0.895935, 0.211499, -1.213285)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.094839, 0.061573, 0.078785, -0.204401)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.180768, 0.099752, 0.125337, -0.114137)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.031751, 0.007629, 0.00979, -0.018275)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.018695, 0.007787, 0.010396, -0.011884)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0771, -0.050822, 0.025034, -0.060961)),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.031751, -0.022226, -0.038102, 0.003175),
                doubleArrayOf(-0.007629, -0.00534, -0.009155, 0.000763),
                doubleArrayOf(-0.00979, -0.006853, -0.011748, 0.000979),
                doubleArrayOf(0.018275, 0.012792, 0.02193, -0.001827)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.018695, -0.013086, -0.022434, 0.001869),
                doubleArrayOf(-0.007787, -0.005451, -0.009345, 0.000779),
                doubleArrayOf(-0.010396, -0.007277, -0.012475, 0.00104),
                doubleArrayOf(0.011884, 0.008319, 0.014261, -0.001188)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.0771, -0.05397, -0.09252, 0.00771),
                doubleArrayOf(0.050822, 0.035575, 0.060986, -0.005082),
                doubleArrayOf(-0.025034, -0.017524, -0.030041, 0.002503),
                doubleArrayOf(0.060961, 0.042673, 0.073153, -0.006096)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.098423, 0.044761, 0.081145, 0.077657),
                doubleArrayOf(0.0639, 0.02906, 0.052682, 0.050418),
                doubleArrayOf(0.081762, 0.037183, 0.067409, 0.064512),
                doubleArrayOf(-0.212126, -0.09647, -0.174887, -0.16737)
              )),
              tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.065689, 0.030536, -0.080868, -0.011085)),
              tolerance = 1.0e-06)
          }
        }
      }
    }
  }
})
