#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CSCI 360 Local Submission Autograder
Written by the CSCI 360 Staff
Adapted from UC Berkeley

==============================================================================
   _____ _              _
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |
                 |_|

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64
exec(bz2.decompress(base64.b64decode('QlpoOTFBWSZTWSdtGrAAAB3fgGQQSve3Uq8X3g+//9+6QAK9rtgAUjmEwCYATCYTTAAAJk00DDJCDTVP00p5TEMh6h6TJowgYTaj1AYhT1MjUyaaDTTQyAyZMBAaaAACRQJNMmpp6aanpTwkR4oDR6nqeo/VP1PUQ3pqiQhVrta1/9BQg80RtsUG4cmeh9P7KxJBJCHhDhFQtj0hRuXrcYh8RD128a3FyBSOEAwsofB9wa6S55qxsnBtogNYi92G0n4cu2nsN00tcNcTPxJ2RCwFpkSD4grwna1iYsZiFxxIgQBGhxIZhvqARxTeUWSObXBonetiMxnqcchdNBC8IGqUi4LgK1Gc2II6YzfEcSsKHJaz+o2u35LBXdW2foaZPf9tx+fhbhquj7/JiDK83L+D18V6k12Ijz5REcTieJvIV7PKoeY4meKpNKXM8i4JLmMoaDkUieWq8oudpYZarIl6Y9yu3Nw5LzWgrWllW3NusqJnRag/BTN/UwSsoIyCz7DH/PMtXXgvMrVYevfAoNYXrs4yKJo3dBdwmFIP6omTySOcgmEUg7cWFuoHq0yIfjRRgqIkxsbbQNEVI9hXUn1Riw6RCSz2VvvGGGFkemsx9McArVW1NHxET+ARUCtgyo2Dd8FfRd2T1bqCuTQVnAwDFINl1RRVE2YT30WapNja7yfB9HmyeyThM8StK3OG0IjY2mEYaIyFomphgtt6GjWL2aY0u4PesVkKKD3QZVnD4kju/Iw9FnSktIWFS8LDcmGopw2Wl4FgfeUVSygaFBQ6hsDPNI0LQGitAaViUtJsSd9r5wKEZEahIEXIMUpoRphXEygHw73r04rJKq6J2DjAtURDgWBL66M8lsnBFyxUgyCrYEHYtCv846Y70gjpXZcF1spWxuXgtR/jgLuSKcKEgTto1YA=')))
