/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

/**
 *
 */
sealed class LayerType {

  /**
   *
   */
  enum class Property {
    Feedforward,
    Recurrent
  }

  /**
   *
   */
  enum class Connection(val property: Property) {
    Feedforward(property = Property.Feedforward),
    SimpleRecurrent(property = Property.Recurrent),
    GRU(property = Property.Recurrent),
    LSTM(property = Property.Recurrent),
    CFN(property = Property.Recurrent),
    RAN(property = Property.Recurrent)
  }

  /**
   *
   */
  enum class Input {
    Dense,
    Sparse,
    SparseBinary
  }
}
