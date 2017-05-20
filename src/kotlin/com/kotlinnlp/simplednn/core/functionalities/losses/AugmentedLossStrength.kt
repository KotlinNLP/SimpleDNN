/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.losses

/**
 *
 */
enum class AugmentedLossStrength(val value: Int, val description: String, val weight: Double) {
  NONE(1, "none", 0.0),
  SOFT(2, "soft", 0.01),
  MEDIUM(3, "medium", 0.1),
  HARD(4, "hard", 1.0)
}
