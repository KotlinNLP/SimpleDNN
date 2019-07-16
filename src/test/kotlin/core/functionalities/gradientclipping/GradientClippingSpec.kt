/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.gradientclipping

import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe


class GradientClippingSpec: Spek({

  describe("the gradient clipping") {

    context("clip at value") {
      val paramsErrors: ParamsErrorsList = GradientClippingUtils.buildErrors()

    }


  }
})
