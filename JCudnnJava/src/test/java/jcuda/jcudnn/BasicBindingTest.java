/*
 * JCuda - Java bindings for CUDA
 *
 * http://www.jcuda.org
 */

package jcuda.jcudnn;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;

/**
 * Utility class for the most basic possible test of a JNI binding: 
 * It offers methods to test whether all public static methods of a 
 * given class can be called without causing an UnsatisfiedLinkError
 */
public class BasicBindingTest
{
    /**
     * Returns whether all public static methods of the given
     * class can be invoked with {@link #testMethod(Method)}
     * 
     * @param c The class
     * @param skipped Names of methods that should be skipped in the test
     * because CUDA causes a segfault that I'm not responsible for...
     * @return Whether the methods can be invoked
     */
    public static boolean testBinding(Class<?> c, Collection<String> skipped)
    {
        logInfo("Testing " + c);

        boolean passed = true;
        int modifiers = Modifier.PUBLIC | Modifier.STATIC;
        Method[] methods = c.getMethods();
        Arrays.sort(methods, new Comparator<Method>()
        {
            @Override
            public int compare(Method m0, Method m1)
            {
                return m0.getName().compareTo(m1.getName());
            }
            
        });
        for (Method method : methods)
        {
            if (skipped.contains(method.getName()))
            {
                logWarning("Skipping test for " + method);
            }
            else
            {
                if ((method.getModifiers() & modifiers) == modifiers)
                {
                    passed &= testMethod(method);
                }
            }
        }
        logInfo("Testing " + c + " done");
        return passed;
    }

    /**
     * Tests whether the given method can be invoked on its declaring
     * class, with default parameters, without causing an 
     * UnsatisfiedLinkError
     * 
     * @param method The method
     * @return Whether the method can be invoked
     */
    private static boolean testMethod(Method method)
    {
        Class<?>[] parameterTypes = method.getParameterTypes();
        Object parameters[] = new Object[parameterTypes.length];
        for (int i = 0; i < parameters.length; i++)
        {
            parameters[i] = getParameterValue(parameterTypes[i]);
        }
        try
        {
            logInfo("Calling " + method);
            method.invoke(method.getDeclaringClass(), parameters);
        }
        catch (IllegalArgumentException e)
        {
            logWarning("IllegalArgumentException for " + method + 
                ": " + e.getMessage());
            e.printStackTrace();
        }
        catch (IllegalAccessException e)
        {
            logWarning("IllegalAccessException for " + method + 
                ": "+e.getMessage());
            e.printStackTrace();
        }
        catch (InvocationTargetException e)
        {
            if (e.getCause() instanceof UnsatisfiedLinkError)
            {
                e.getCause().printStackTrace();
                logWarning("Missing " + method);
                return false;
            }
            //logWarning("InvocationTargetException for " + method + 
            //    ": "+e.getMessage());
            //e.printStackTrace();
        }
        return true;
    }

    /**
     * Returns the default value for a parameter with the given type
     * 
     * @param parameterType The type
     * @return The default value
     */
    private static Object getParameterValue(Class<?> parameterType)
    {
        if (!parameterType.isPrimitive())
        {
            return null;
        }
        if (parameterType == byte.class)
        {
            return (byte) 0;
        }
        if (parameterType == char.class)
        {
            return (char) 0;
        }
        if (parameterType == short.class)
        {
            return (short) 0;
        }
        if (parameterType == int.class)
        {
            return (int) 0;
        }
        if (parameterType == long.class)
        {
            return (long) 0;
        }
        if (parameterType == float.class)
        {
            return (float) 0;
        }
        if (parameterType == double.class)
        {
            return (double) 0;
        }
        if (parameterType == boolean.class)
        {
            return false;
        }
        return null;
    }

    private static final boolean LOG_INFO = false;
    private static final boolean LOG_WARNING = true;

    /**
     * Utility method for info logs
     * 
     * @param message The message
     */
    private static void logInfo(String message)
    {
        if (LOG_INFO)
        {
            System.out.println(message);
        }
    }

    /**
     * Utility method for warning logs
     * 
     * @param message The message
     */
    private static void logWarning(String message)
    {
        if (LOG_WARNING)
        {
            System.out.println(message);
        }
    }

}
