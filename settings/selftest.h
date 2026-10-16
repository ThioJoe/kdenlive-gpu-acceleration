/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

#pragma once

/*
 * The check of the page in Kdenlive itself, in a variant of the page built
 * for the Windows workflow (see selftest.cpp).
 */

class KConfigDialog;
class KPageWidgetItem;
class QLabel;

/** Starts the check, once the application exists */
void selftestStart();

/** Whether to add the page: with GPUFX_SELFTEST_PAGE=0, the check shows the
 *  dialog without it */
bool selftestWantsPage();

/** Shows the page, or the dialog alone when item and status are null, then
 *  saves the report and the screenshot and ends the application */
void selftestReport(KConfigDialog *dialog, KPageWidgetItem *item, QLabel *status);
