/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.utils.progressindicator

import java.io.OutputStream
import java.lang.Math.floor

/**
 *
 */
abstract class ProgressIndicator(protected val total: Int, protected val outputStream: OutputStream) {

  /**
   *
   */
  private var current: Int = 0

  /**
   *
   */
  protected var perc: Int = -1

  /**
   *
   */
  fun tick(amount: Int = 1) {

    this.current += amount

    val curPerc = floor(100.0 * this.current / this.total).toInt()

    if (curPerc > this.perc) {
      this.perc = curPerc
      this.print()
    }
  }

  /**
   *
   */
  private fun print() {

    outputStream.write("\r%s%s".format(
      this.getProgressString(),
      if (this.perc == 100) "\n" else "").toByteArray())

    outputStream.flush()
  }

  /**
   *
   */
  abstract protected fun getProgressString(): String
}
