/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.feedforward.convolution

import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
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
class ConvolutionLayerStructureSpec : Spek({

  describe("a ConvolutionLayerStructure"){

    context("forward") {

      on("input size 4 x 4 (tanh)") {

        val layer = ConvolutionLayerStructureUtils.buildLayer443()
        layer.forward()

        it("should match the expected output for the first channel") {
          assertTrue {
            layer.outputArrays[0].values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.591519, -0.500520, 0.309506),
                    doubleArrayOf(-0.206966, 0.158649, 0.079829),
                    doubleArrayOf(-0.430084, -0.272905, 0.216518))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected output for the second channel") {
          assertTrue {
            layer.outputArrays[1].values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.716297, -0.757362, 0.903324),
                    doubleArrayOf(0.623065, 0.244918, 0.282134),
                    doubleArrayOf(0.861723, -0.859126, 0.926061))),
                tolerance = 1.0e-06)
          }
        }


      }
    }

    on("input size 4 x 4 (tanh), stride 2") {

      val layer = ConvolutionLayerStructureUtils.buildLayer443str2()
      layer.forward()

      it("should match the expected output for the first channel") {
        assertTrue {
          layer.outputArrays[0].values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                  doubleArrayOf(-0.591519, 0.309506),
                  doubleArrayOf(-0.430084, 0.216518))),
              tolerance = 1.0e-06)
        }
      }

      it("should match the expected output for the second channel") {
        assertTrue {
          layer.outputArrays[1].values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                  doubleArrayOf(-0.716297, 0.903324),
                  doubleArrayOf(0.861723, 0.926061))),
              tolerance = 1.0e-06)
        }
      }


    }

    context("backward") {

      on("input size 4 x 4 (tanh)") {

        val layer = ConvolutionLayerStructureUtils.buildLayer443()
//        val paramsErrors = ConvolutionLayerParameters(kernelSize = Shape(2, 2),
//            inputChannels = 3,
//            outputChannels = 2)

        layer.forward()

        layer.outputArrays[0].assignErrors(layer.outputArrays[0].values.sub(
            ConvolutionLayerStructureUtils.getOutputGold1()))
        layer.outputArrays[1].assignErrors(layer.outputArrays[1].values.sub(
            ConvolutionLayerStructureUtils.getOutputGold2()))

        val paramsErrors = layer.backward(propagateToInput = true)
        val params = layer.params

        it("should match the expected errors of the output channel 1") {
          assertTrue {
            layer.outputArrays[0].errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-1.034654, -0.375129, 0.279857),
                    doubleArrayOf(-1.155265, 0.154655, -0.914306),
                    doubleArrayOf(-1.165558, -0.252579, 0.206367))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the output channel 2") {
          assertTrue {
            layer.outputArrays[1].errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.348777, -0.322941, -0.017788),
                    doubleArrayOf(0.381184, -0.709787, 0.259676),
                    doubleArrayOf(-0.035597, -0.225006, 0.131880))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of kernel for input 1 to output 1") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[0])!!.values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-1.693899, -0.110008),
                    doubleArrayOf(-1.028675, -1.511484))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of kernel for input 2 to output 1") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[1])!!.values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-1.252652, 0.368623),
                    doubleArrayOf(-1.551219, 0.022600))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of kernel for input 3 to output 1") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[2])!!.values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.665568, -0.185757),
                    doubleArrayOf(1.555463, -0.891677))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of kernel for input 1 to output 2") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[3])!!.values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.333782, -0.245542),
                    doubleArrayOf(0.443482, -0.651037))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of kernel for input 2 to output 2") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[4])!!.values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.522095,-0.138857),
                    doubleArrayOf(0.239587, 0.310294))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of kernel for input 3 to output 2") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[5])!!.values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.044085, -0.114367),
                    doubleArrayOf(-0.293713, 0.777384))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of bias 1") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[6])!!.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-4.256613)),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of bias 2") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[7])!!.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.887157)),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input 1") {
          assertTrue {
            layer.inputArrays[0].errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.587082, -0.302388, -0.474968, 0.071505),
                    doubleArrayOf(0.361884, 0.273068, 0.275807, -0.104494),
                    doubleArrayOf(0.742372, -0.714749, 0.364375, 0.011948),
                    doubleArrayOf(-0.014238, 0.058590, 0.280516, -0.139329))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input 2") {
          assertTrue {
            layer.inputArrays[1].errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.413861, 0.253517, -0.074430, -0.027985),
                    doubleArrayOf(0.323763, -1.261232, -0.251799, 0.327292),
                    doubleArrayOf(0.388815, -0.534598, -0.622371, -0.609803),
                    doubleArrayOf(-0.120116, -1.128798, -0.396003, 0.304423))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input 3") {
          assertTrue {
            layer.inputArrays[2].errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.516159, -0.681568, -0.105977, 0.104827),
                    doubleArrayOf(-1.185331, -0.408733, -0.905697, -0.212995),
                    doubleArrayOf(-1.418762, -0.281844, -0.920785, 0.056308),
                    doubleArrayOf(-0.568540, -0.283637, -0.090086, 0.094025))),
                tolerance = 1.0e-06)
          }
        }


      }

      on("input size 4 x 4 (tanh) stride 2") {

        val layer = ConvolutionLayerStructureUtils.buildLayer443str2()
//        val paramsErrors = ConvolutionLayerParameters(kernelSize = Shape(2, 2),
//            inputChannels = 3,
//            outputChannels = 2)

        layer.forward()

        layer.outputArrays[0].assignErrors(layer.outputArrays[0].values.sub(
            ConvolutionLayerStructureUtils.getOutputGold3()))
        layer.outputArrays[1].assignErrors(layer.outputArrays[1].values.sub(
            ConvolutionLayerStructureUtils.getOutputGold4()))

        val paramsErrors = layer.backward(propagateToInput = true)
        val params = layer.params

        it("should match the expected errors of the output channel 1") {
          assertTrue {
            layer.outputArrays[0].errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-1.034654, 0.279857),
                    doubleArrayOf(-1.165558, 0.206367))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the output channel 2") {
          assertTrue {
            layer.outputArrays[1].errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.348777, 0.166215),
                    doubleArrayOf(0.221836, -0.010529))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of kernel for input 1 to output 1") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[0])!!.values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-1.474360, -0.332049),
                    doubleArrayOf(0.500471, -0.319071))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of kernel for input 2 to output 1") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[1])!!.values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.973070, 0.585176),
                    doubleArrayOf(-0.671462, 0.425471))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of kernel for input 3 to output 1") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[2])!!.values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.432622, -1.188760),
                    doubleArrayOf(1.017080, 0.569278))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of kernel for input 1 to output 2") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[3])!!.values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.117954, -0.080989),
                    doubleArrayOf(0.293912, -0.174576))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of kernel for input 2 to output 2") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[4])!!.values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.115032, 0.134828),
                    doubleArrayOf(0.113198, 0.354131))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of kernel for input 3 to output 2") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[5])!!.values.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.333472, -0.026474),
                    doubleArrayOf(-0.010793, 0.411303))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of bias 1") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[6])!!.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.713987)),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of bias 2") {
          assertTrue {
            paramsErrors.getErrorsOf(params.paramsList[7])!!.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.028744)),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input 1") {
          assertTrue {
            layer.inputArrays[0].errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.587083, -0.554541, -0.173172, 0.200308),
                    doubleArrayOf(-0.139511, 0.417366, 0.066486, -0.177580),
                    doubleArrayOf(0.538412, -0.194382, -0.101078, 0.054540),
                    doubleArrayOf(0.088734, -0.083097, -0.004212, -0.011160))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input 2") {
          assertTrue {
            layer.inputArrays[1].errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.413862, 0.103465, -0.111943, -0.027986),
                    doubleArrayOf(-0.138343, -1.245089, 0.044607, 0.401466),
                    doubleArrayOf(0.466223, 0.116556, -0.082547, -0.020637),
                    doubleArrayOf(-0.094372, -0.849350, 0.019584, 0.176254))),
                tolerance = 1.0e-06)
          }
        }

        it("should match the expected errors of the input 3") {
          assertTrue {
            layer.inputArrays[2].errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.516159, -0.553373, 0.118050, 0.178429),
                    doubleArrayOf(-0.377816, -0.346442, 0.073443, 0.122458),
                    doubleArrayOf(-0.765886, -0.377489, 0.126979, 0.078335),
                    doubleArrayOf(-0.671513, -0.144377, 0.107396, 0.037062))),
                tolerance = 1.0e-06)
          }
        }


      }
    }
  }
})