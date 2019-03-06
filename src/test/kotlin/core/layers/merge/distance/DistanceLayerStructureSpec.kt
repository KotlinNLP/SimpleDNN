/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.distance

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import core.layers.merge.distance.DistanceLayerUtils
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertTrue

/**
 *
 */
class DistanceLayerStructureSpec: Spek({

  describe("a SumLayer")
  {

    on("forward") {

      val layer = DistanceLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.06081)),
              tolerance = 1.0e-05)
        }
      }
    }
  }
})