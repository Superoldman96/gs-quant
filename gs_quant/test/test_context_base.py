"""
Copyright 2019 Goldman Sachs.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

import asyncio
import threading

import pytest

from gs_quant import context_base
from gs_quant.context_base import ContextBase, ContextBaseWithDefault
from gs_quant.errors import MqUninitialisedError, MqValueError


class SimpleContext(ContextBase):
    def __init__(self, value=None):
        self.value = value


class DefaultContext(ContextBaseWithDefault):
    def __init__(self, value='default'):
        self.value = value


@pytest.fixture(autouse=True)
def reset_context_state():
    """The context stack and is_entered flags live in module-global ContextVars keyed by class name,
    so sync tests share state. Reset it after each test to keep tests isolated and order-independent."""
    yield
    for cls in (SimpleContext, DefaultContext):
        context_base._get_context_var('{}_path'.format(cls.__name__)).set(())
        context_base._get_context_var('{}_default'.format(cls.__name__)).set(None)
    context_base._entered_instances.set(frozenset())


class TestContextStack:
    def test_push_pop_current(self):
        ctx = SimpleContext(value='a')
        with ctx:
            assert SimpleContext.current is ctx
            assert SimpleContext.current.value == 'a'
        assert not SimpleContext.current_is_set

    def test_nested_contexts(self):
        outer = SimpleContext(value='outer')
        inner = SimpleContext(value='inner')
        with outer:
            assert SimpleContext.current.value == 'outer'
            assert not SimpleContext.has_prior
            with inner:
                assert SimpleContext.current.value == 'inner'
                assert SimpleContext.has_prior
                assert SimpleContext.prior is outer
            assert SimpleContext.current.value == 'outer'

    def test_path(self):
        a = SimpleContext(value='a')
        b = SimpleContext(value='b')
        assert SimpleContext.path == ()
        with a:
            assert SimpleContext.path == (a,)
            with b:
                assert SimpleContext.path == (a, b) or SimpleContext.path == (b, a)
                assert SimpleContext.path[0] is b  # most recent is first
        assert SimpleContext.path == ()

    def test_current_uninitialized_raises(self):
        with pytest.raises(MqUninitialisedError):
            _ = SimpleContext.current

    def test_current_setter(self):
        ctx = SimpleContext(value='set')
        SimpleContext.current = ctx
        assert SimpleContext.current is ctx
        # Clean up
        SimpleContext.pop()

    def test_current_setter_blocked_in_nested(self):
        outer = SimpleContext(value='outer')
        inner = SimpleContext(value='inner')
        with outer:
            with inner:
                with pytest.raises(MqValueError):
                    SimpleContext.current = SimpleContext(value='nope')


class TestIsEntered:
    def test_not_entered_by_default(self):
        ctx = SimpleContext()
        assert not ctx.is_entered

    def test_entered_inside_with(self):
        ctx = SimpleContext()
        with ctx:
            assert ctx.is_entered
        assert not ctx.is_entered

    def test_entered_after_exception(self):
        ctx = SimpleContext()
        try:
            with ctx:
                assert ctx.is_entered
                raise ValueError('test')
        except ValueError:
            pass
        assert not ctx.is_entered


class TestDefaultContext:
    def test_default_value(self):
        assert DefaultContext.current_is_set
        assert DefaultContext.current.value == 'default'

    def test_override_default(self):
        ctx = DefaultContext(value='custom')
        with ctx:
            assert DefaultContext.current.value == 'custom'
        assert DefaultContext.current.value == 'default'


class TestAsyncIsolation:
    @pytest.mark.asyncio
    async def test_concurrent_tasks_isolated(self):
        """Two concurrent asyncio tasks should each see their own context stack."""
        entered = {name: asyncio.Event() for name in ('A', 'B')}
        read = {name: asyncio.Event() for name in ('A', 'B')}
        results = {}

        async def task(name, other, value):
            ctx = SimpleContext(value=value)
            async with ctx:
                entered[name].set()
                await entered[other].wait()
                results[name] = SimpleContext.current.value
                read[name].set()
                await read[other].wait()

        await asyncio.gather(
            asyncio.create_task(task('A', 'B', 'value_a')),
            asyncio.create_task(task('B', 'A', 'value_b')),
        )

        assert results['A'] == 'value_a'
        assert results['B'] == 'value_b'

    @pytest.mark.asyncio
    async def test_nested_async_contexts(self):
        outer = SimpleContext(value='outer')
        inner = SimpleContext(value='inner')
        async with outer:
            assert SimpleContext.current.value == 'outer'
            async with inner:
                assert SimpleContext.current.value == 'inner'
            assert SimpleContext.current.value == 'outer'

    @pytest.mark.asyncio
    async def test_async_is_entered(self):
        ctx = SimpleContext()
        assert not ctx.is_entered
        async with ctx:
            assert ctx.is_entered
        assert not ctx.is_entered


class TestThreadIsolation:
    def test_different_threads_isolated(self):
        """Different threads should have independent context stacks."""
        barrier = threading.Barrier(2)
        results = {}

        def thread_fn(name, value):
            ctx = SimpleContext(value=value)
            with ctx:
                barrier.wait()
                results[name] = SimpleContext.current.value
                barrier.wait()

        t1 = threading.Thread(target=thread_fn, args=('T1', 'val1'))
        t2 = threading.Thread(target=thread_fn, args=('T2', 'val2'))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results['T1'] == 'val1'
        assert results['T2'] == 'val2'

    def test_is_entered_isolated_across_threads(self):
        """is_entered should be per-thread, not shared across threads."""
        ctx = SimpleContext(value='shared')
        barrier = threading.Barrier(2)
        results = {}

        def thread_fn(name, enter):
            if enter:
                with ctx:
                    barrier.wait()
                    results[name] = ctx.is_entered
                    barrier.wait()
            else:
                barrier.wait()
                results[name] = ctx.is_entered
                barrier.wait()

        t1 = threading.Thread(target=thread_fn, args=('entered', True))
        t2 = threading.Thread(target=thread_fn, args=('not_entered', False))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results['entered'] is True
        assert results['not_entered'] is False

    @pytest.mark.asyncio
    async def test_is_entered_isolated_across_tasks(self):
        """is_entered should be per-task, not shared across async tasks."""
        ctx = SimpleContext(value='shared')
        entered_event = asyncio.Event()
        checked_event = asyncio.Event()
        results = {}

        async def entering_task():
            async with ctx:
                entered_event.set()
                await checked_event.wait()
                results['entered'] = ctx.is_entered

        async def checking_task():
            await entered_event.wait()
            results['not_entered'] = ctx.is_entered
            checked_event.set()

        await asyncio.gather(
            asyncio.create_task(entering_task()),
            asyncio.create_task(checking_task()),
        )

        assert results['entered'] is True
        assert results['not_entered'] is False
