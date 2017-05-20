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
class ProgressIndicatorBar(total: Int, outputStream: OutputStream = System.out, val barLength: Int = 50) :
  ProgressIndicator(total = total, outputStream = outputStream) {

  /**
   *
   */
  override fun getProgressString(): String {

    var printStr = "|"

    val progressLength = Math.floor(barLength * this.perc / 100.0).toInt()

    (0 until progressLength).forEach({ printStr += "█" })
    (progressLength until barLength).forEach({ printStr += " " })

    printStr += "| ${this.perc}%"

    return printStr
  }

}
