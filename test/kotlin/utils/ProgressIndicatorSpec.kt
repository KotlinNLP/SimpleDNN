/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package utils

import com.kotlinnlp.simplednn.utils.progressindicator.*
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import java.io.ByteArrayOutputStream
import kotlin.test.assertEquals

/**
 *
 */
class ProgressIndicatorSpec: Spek({

  describe("a ProgressIndicatorBar") {

    context("total = 10, barLength = default (50)") {

      val outputStream = ByteArrayOutputStream()
      val progress = ProgressIndicatorBar(total = 10, outputStream = outputStream)

      on("first tick") {

        progress.tick()

        it("should print the expected string") {
          assertEquals(
            "\r|█████                                             | 10%",
            String(outputStream.toByteArray()))
        }
      }
    }

    context("total = 10, barLength = 20") {

      val outputStream = ByteArrayOutputStream()
      val progress = ProgressIndicatorBar(total = 10, outputStream = outputStream, barLength = 20)

      on("first tick") {

        progress.tick()

        it("should print the expected string") {
          assertEquals("\r|██                  | 10%", String(outputStream.toByteArray()))
        }
      }

      on("second tick") {

        progress.tick()

        it("should print the expected string") {
          assertEquals(
            "\r|██                  | 10%" +
            "\r|████                | 20%",
            String(outputStream.toByteArray()))
        }
      }
    }
  }

  describe("a ProgressIndicatorPercentage") {

    val outputStream = ByteArrayOutputStream()
    val progress = ProgressIndicatorPercentage(total = 10, outputStream = outputStream)

    on("first tick") {

      progress.tick()

      it("should print the expected string") {
        assertEquals("\r[10%]", String(outputStream.toByteArray()))
      }
    }

    on("last tick") {

      (1 until 10).forEach({ progress.tick() })

      it("should print the expected string") {
        assertEquals(
          "\r[10%]" +
          "\r[20%]" +
          "\r[30%]" +
          "\r[40%]" +
          "\r[50%]" +
          "\r[60%]" +
          "\r[70%]" +
          "\r[80%]" +
          "\r[90%]" +
          "\r[100%]\n",
          String(outputStream.toByteArray()))
      }
    }
  }

  describe("a ProgressIndicatorIcon") {

    val outputStream = ByteArrayOutputStream()
    val progress = ProgressIndicatorIcon(total = 10, outputStream = outputStream)

    on("first tick") {

      progress.tick()

      it("should print the expected string") {
        assertEquals("\r-", String(outputStream.toByteArray()))
      }
    }

    on("second tick") {

      progress.tick()

      it("should print the expected string") {
        assertEquals(
          "\r-" +
          "\r\\",
          String(outputStream.toByteArray()))
      }
    }

    on("last tick") {

      (2 until 10).forEach({ progress.tick() })

      it("should print the expected string") {
        assertEquals(
          "\r-" +
          "\r\\" +
          "\r|" +
          "\r/" +
          "\r-" +
          "\r\\" +
          "\r|" +
          "\r/" +
          "\r-" +
          "\r\\\n"
          , String(outputStream.toByteArray()))
      }
    }
  }
})
