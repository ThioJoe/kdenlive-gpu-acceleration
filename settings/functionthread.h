/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QThread>

#include <functional>

/** Runs a function in a thread of its own, which deletes itself when done.
 *  Not QThread::create(), which hands Qt a std::future: the page has its own
 *  copy of the C++ runtime, whose objects Kdenlive's Qt must not use (a
 *  thread waiting on such a future hangs in Kdenlive). */
class FunctionThread : public QThread
{
public:
    explicit FunctionThread(std::function<void()> function)
        : m_function(std::move(function))
    {
        connect(this, &QThread::finished, this, &QObject::deleteLater);
    }

protected:
    void run() override
    {
        m_function();
    }

private:
    std::function<void()> m_function;
};
