/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

/**
 * The layer type.
 */
sealed class LayerType {

  /**
   * The property of a layer connection.
   */
  enum class Property {
    Batch,
    Merge,
    Feedforward,
    Recurrent
  }

  /**
   * The layer connection type.
   */
  enum class Connection(val property: Property) {
    Feedforward(property = Property.Feedforward),
    Highway(property = Property.Feedforward),
    BatchNorm(property = Property.Batch),
    SquaredDistance(property = Property.Feedforward),
    Affine(property = Property.Merge),
    Biaffine(property = Property.Merge),
    Concat(property = Property.Merge),
    ConcatFeedforward(property = Property.Merge),
    Sum(property = Property.Merge),
    Sub(property = Property.Merge),
    Avg(property = Property.Merge),
    Product(property = Property.Merge),
    SimpleRecurrent(property = Property.Recurrent),
    GRU(property = Property.Recurrent),
    LSTM(property = Property.Recurrent),
    CFN(property = Property.Recurrent),
    RAN(property = Property.Recurrent),
    DeltaRNN(property = Property.Recurrent),
    IndRNN(property = Property.Recurrent),
    LTM(property = Property.Recurrent),
    TPR(property = Property.Recurrent)
  }

  /**
   * The layer input type.
   */
  enum class Input {
    Dense,
    Sparse,
    SparseBinary
  }
}
