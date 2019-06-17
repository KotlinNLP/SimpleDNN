/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.tpr

import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr.TPRLayer
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


      }

      on("with previous state only") {


      }

      on("with init hidden") {


      }

      on("with next state only") {


      }

      on("with previous and next state") {


      }
    }
  }
})