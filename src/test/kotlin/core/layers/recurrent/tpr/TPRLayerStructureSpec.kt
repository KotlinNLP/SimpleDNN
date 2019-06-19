/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.tpr

import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr.TPRLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr.TPRLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

class TPRLayerStructureSpec: Spek({

  describe("a TPRLayer") {

    context("forward") {

      on("without previous state context") {

        val layer = TPRLayerStructureUtils.buildLayer(TPRLayerContextWindow.Empty())
        layer.forward()

        it("should match the expected Symbol Attention Vector") {
          assertTrue {
            layer.aS.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.569546, 0.748381, 0.509998, 0.345246)),
                tolerance = 0.005)
          }
        }

        it("should match the expected Role Attention Vector") {
          assertTrue {
            layer.aR.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.291109, 0.391740, 0.394126)),
                tolerance = 0.005)
          }
        }

        it("should match the expected Symbol Vector") {
          assertTrue {
            layer.s.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.142810, 0.913446, 0.425346)),
                tolerance = 0.005)
          }
        }

        it("should match the expected Role Vector") {
          assertTrue {
            layer.r.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.352204, 0.205093)),
                tolerance = 0.005)
          }
        }

        it("should match the expected Output vector") {
          assertTrue {
            layer.outputArray.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.050298, 0.029289, 0.321719, 0.187342, 0.149808, 0.087235)),
                tolerance = 0.005)
          }
        }
      }

      on("with previous state context") {

        val layer = TPRLayerStructureUtils.buildLayer(TPRLayerContextWindow.Back())
        layer.forward()

        it("should match the expected Symbol Attention Vector") {
          assertTrue {
            layer.aS.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.310421, 0.880352, 0.356117, 0.575599)),
                tolerance = 0.005)
          }
        }

        it("should match the expected Role Attention Vector") {
          assertTrue {
            layer.aR.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.075481, 0.619886, 0.357379)),
                tolerance = 0.005)
          }
        }

        it("should match the expected Symbol Vector") {
          assertTrue {
            layer.s.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.169241, 0.635193, 0.132934)),
                tolerance = 0.005)
          }
        }

        it("should match the expected Role Vector") {
          assertTrue {
            layer.r.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.323372, 0.182359)),
                tolerance = 0.005)
          }
        }

        it("should match the expected Output vector") {
          assertTrue {
            layer.outputArray.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.054727, 0.030862, 0.205404, 0.115833, 0.042987, 0.024241)),
                tolerance = 0.005)
          }
        }
      }

    }

    context("backward") {

      on("without previous and next state") {

        val layer = TPRLayerStructureUtils.buildLayer(TPRLayerContextWindow.Empty())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
            output = layer.outputArray.values,
            outputGold = TPRLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params as TPRLayerParameters

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.519071, -0.720710, 0.471719, -1.452657, -0.300191, -0.022764)),
                tolerance = 0.005)
          }
        }

        it("should match the expected errors of the Binding Matrix") {
          assertTrue {
            layer.bindingMatrix.errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.519071, -0.720710),
                    doubleArrayOf(0.471719, -1.452657),
                    doubleArrayOf(-0.300191, -0.022764)
                )),
                tolerance = 0.005)
          }
        }

        it("should match the expected errors of the Symbol Vector") {
          assertTrue {
            layer.s.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.330854, -0.131789, -0.110397)),
                tolerance = 0.005)
          }
        }

        it("should match the expected errors of the Role Vector") {
          assertTrue {
            layer.r.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.228986, -1.439532)),
                tolerance = 0.005)
          }
        }

        it("should match the expected errors of the Symbol attention Vector") {
          assertTrue {
            layer.aS.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.054415529690289, -0.009069085833328, -0.021993209302265, -0.004547012736255)),
                tolerance = 0.005)
          }
        }

        it("should match the expected errors of the Role attention Vector") {
          assertTrue {
            layer.aR.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.070218831996901,-0.052233526362676, -0.017970777983179)),
                tolerance = 0.005)
          }
        }


        it("should match the expected errors of the Symbols embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.S)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.188436752484536, -0.247605225547889, -0.168735193947239, -0.114226262865736),
                    doubleArrayOf(-0.075060214536815, -0.098628856128193, -0.06721247150887, -0.045499870292789),
                    doubleArrayOf(-0.062876419465944, -0.082619392545444, -0.05630252428684, -0.038114318588475)
                )),
                tolerance = 0.005)
          }
        }

        it("should match the expected errors of the Roles embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.R)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.06666014408481, 0.089703290625912, 0.090249505767454),
                    doubleArrayOf(-0.419062189806341, -0.563924034648095, -0.567357842307266)
                )),
                tolerance = 0.005)
          }
        }


        it("should match the expected errors of the input -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf( 0.043532423752231, 0.04897397672126, -0.04897397672126, -0.005441552969029),
                    doubleArrayOf(0.007255268666663, 0.008162177249996, -0.008162177249996, -0.000906908583333),
                    doubleArrayOf(0.017594567441812, 0.019793888372039, -0.019793888372039, -0.002199320930227),
                    doubleArrayOf(0.003637610189004, 0.00409231146263, -0.00409231146263, -0.000454701273626)
                )),
                tolerance = 0.005)
          }
        }

        it("should match the expected errors of the input -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.056175065597521, 0.063196948797211, -0.063196948797211, -0.00702188319969),
                    doubleArrayOf(0.041786821090141, 0.047010173726408, -0.047010173726408, -0.005223352636268),
                    doubleArrayOf(0.014376622386543, 0.016173700184861, -0.016173700184861, -0.001797077798318)
                )),
                tolerance = 0.005)
          }
        }

        it("should match the expected errors of the recurrent -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                )),
                tolerance = 0.005)
          }
        }

        it("should match the expected errors of the recurrent -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                )),
                tolerance = 0.005)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.083094202621031, -0.079789163746023, -0.000560627296261, 0.023229550127503)),
                tolerance = 0.005)
          }
        }

      }

      on("with previous state only") {


      }

      on("with next state only") {


      }

      on("with previous and next state") {


      }
    }
  }
})