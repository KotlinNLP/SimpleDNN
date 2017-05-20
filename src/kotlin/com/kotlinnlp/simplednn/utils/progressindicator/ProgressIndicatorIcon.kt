/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.utils.progressindicator

import java.io.OutputStream

/**
 *
 */
class ProgressIndicatorIcon(total: Int, outputStream: OutputStream = System.out) :
  ProgressIndicator(total = total, outputStream = outputStream) {

  /**
   *
   */
  private val iconSequence = arrayOf('-', '\\', '|', '/', '-', '\\', '|', '/')

  /**
   *
   */
  private var iconIndex: Int = 0

  /**
   * 
   */
  override fun getProgressString(): String {

    val printStr = "${iconSequence[this.iconIndex]}"
    this.iconIndex = (this.iconIndex + 1) % iconSequence.size

    return printStr
  }

}
