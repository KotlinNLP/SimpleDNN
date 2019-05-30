/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.optimizer

import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class IterableParamsSpec : Spek({

  describe("an IterableParams") {

    context("Dense params") {

      context("assignValues") {

        val params1 = IterableParamsUtils.buildDenseParams1()
        val params2 = IterableParamsUtils.buildDenseParams2()

        params1.assignValues(params2)

        it("should assign the expected values to the first parameters") {
          assertTrue {
            (params2.unit.weights.values)
              .equals(params1.unit.weights.values, tolerance = 1.0e-06)
          }
        }

        it("should assign the expected values to the second parameters") {
          assertTrue {
            params2.unit.biases.values.equals(params1.unit.biases.values, tolerance = 1.0e-06)
          }
        }
      }
    }
  }
})
