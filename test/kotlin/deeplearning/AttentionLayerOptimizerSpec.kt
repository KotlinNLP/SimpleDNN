/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionlayer.AttentionLayerOptimizer
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.attentionlayer.AttentionLayerStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import deeplearning.utils.AttentionLayerUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class AttentionLayerOptimizerSpec : Spek({

  describe("an AttentionLayerOptimizer") {

    on("accumulation") {

      val attentionLayer = AttentionLayerUtils.buildAttentionLayer()
      val inputSequence = AttentionLayerUtils.buildInputSequence()
      val structure = AttentionLayerStructure(
        inputSequence = inputSequence,
        attentionSequence = AttentionLayerUtils.buildAttentionSequence(inputSequence))
      val optimizer = AttentionLayerOptimizer(
        attentionLayer = attentionLayer,
        updateMethod = LearningRateMethod(learningRate = 0.1))

      structure.contextVectorErrors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.2, -0.5))
      optimizer.accumulateErrors(structure)

      structure.contextVectorErrors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, -0.3))
      optimizer.accumulateErrors(structure)

      optimizer.update()

      it("should update the context vector correctly") {
        assertTrue {
          attentionLayer.contextVector.values.equals(
            DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.33, -0.46)),
            tolerance = 1.0e-06
          )
        }
      }
    }
  }
})
