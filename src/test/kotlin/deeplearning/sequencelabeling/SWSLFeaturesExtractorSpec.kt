/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.sequencelabeling

import com.kotlinnlp.simplednn.deeplearning.sequencelabeling.SWSLFeaturesExtractor
import com.kotlinnlp.simplednn.deeplearning.sequencelabeling.SWSLabeler
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import deeplearning.sequencelabeling.utils.SWSLNetworkUtils
import deeplearning.sequencelabeling.utils.SlidingWindowSequenceUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 *
 */
class SWSLFeaturesExtractorSpec: Spek({

  describe("an SWSLFeaturesExtractor") {

    context("input=2, hidden=5, output=10, prevWindow=3, nextWindow=3, labelEmbedding=2, sequence=5, focus=3") {

      val sequence = SlidingWindowSequenceUtils.buildSlidingWindowSequence()
      sequence.setFocus(1)

      val featuresExtractor = SWSLFeaturesExtractor(
        sequence = sequence,
        labels = arrayListOf(SWSLabeler.Label(0)),
        network = SWSLNetworkUtils.buildNetwork())

      on("getFeatures") {

        val features = featuresExtractor.getFeatures()

        it("should return features of shape 20x1") {
          assertEquals(Shape(20, 1), features.shape)
        }

        it("should return the expected precomputed features") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(doubleArrayOf(
              0.0, 0.0, 0.0, 0.0, 0.4, -0.9, // left context labels
              0.0, 0.0, 0.0, 0.0, 10.0, 11.0, // left context
              20.0, 21.0, // focus element
              30.0, 31.0, 40.0, 41.0, 50.0, 51.0 // right context
            )).equals(features, tolerance = 1.0e-06)
          }
        }
      }
    }
  }
})
