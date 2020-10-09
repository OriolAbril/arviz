(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("83b48f8d-ac61-48a6-9b87-4ffc415427b6");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '83b48f8d-ac61-48a6-9b87-4ffc415427b6' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"d1fa4840-9af4-4734-923e-37a6222d9e0b":{"roots":{"references":[{"attributes":{},"id":"5286","type":"BasicTickFormatter"},{"attributes":{"overlay":{"id":"5261"}},"id":"5255","type":"BoxZoomTool"},{"attributes":{},"id":"5259","type":"SaveTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"5261","type":"BoxAnnotation"},{"attributes":{"source":{"id":"5272"}},"id":"5276","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5279","type":"Line"},{"attributes":{},"id":"5284","type":"BasicTickFormatter"},{"attributes":{"formatter":{"id":"5286"},"ticker":{"id":"5250"}},"id":"5249","type":"LinearAxis"},{"attributes":{"source":{"id":"5277"}},"id":"5281","type":"CDSView"},{"attributes":{"axis":{"id":"5249"},"dimension":1,"ticker":null},"id":"5252","type":"Grid"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5278","type":"Line"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"5262","type":"PolyAnnotation"},{"attributes":{},"id":"5256","type":"WheelZoomTool"},{"attributes":{},"id":"5250","type":"BasicTicker"},{"attributes":{},"id":"5292","type":"Selection"},{"attributes":{"data":{"x":{"__ndarray__":"bso/3qeR+b9DttS5fGP5v7IxN3Y16vW/cryB81XY9L/lpFpwVeLzv4KkDvjZLPO/VTzZVe6s8b8d3/OlLqzxv379gz2/DPC/6MX+Wyvr7b8AaaihV6/tv7OxDMDb5uy/W2cr6Sqz7L8Sfs6UXvfqvzm5jS2Ezeq/L5lOUO6u6r9uoHjjqQ/nv2AIfSz14+W/6vt7XbXi5L99d/C3/bbkv/nYxMaXguK/bIyAxpOB4r9DQWqpw1PevzooaNnCYNy/vBzYgFua279PH+ueYcnav4Y0S58rVdm/qutNN9a72L/0H/i7yC3Wv2Ns7qNu79O/J/PNPsLQ07/Gq5QhAnTSv8AppITXbdK/eaz4wcXf0b/ETRGk1CHRv4ELk5inpdC/OSbU0hsW0L9iMo//JvHOv9nu7H3vtsy/bnPHxZ3Ky79UVKUypinKv6A72DElQMe//WRdtxTywr8xGM5Zm3DCv95tFkQJyb+/uxI23MWbv789QRJsL++4v14BvxLhi7K/Pz2fncrDnr/Q2Ln0eseMv9MvIE2uAXu/d/jH6dhddj9+p/NBCfeyP69qYEfZEcQ/RyxayvkUxD+5cX98tB/JP8mvCK0tmc0/9K5QonPczT9BxCuJ4GPQP5Ga47RIZNE/TBNZPZwf0j/oQa8Vb7bUP7bsmCkcmNU/RefcMjOm1T/nPr/g5C/ZP3p8kYxTjto/B5pLJKs03T/S6rKcIHTdP+BYVOKbPeA/LizMQOgD4T8y+wd70RPiP4R/pvkAMuM/pS8Eyn065D8rgksDDcjkP6A6cl8R7eQ/nZl0gwI05T/X9uAhUqToP2Txgy13aOk/1I+hkavf6j9LaGhVCBrrP+fkoEaZa+s/5iQRyWk17z9UpWPrVFLvP8/HGSbcp/A/259ZNl7E8D87+odOgLPxP26CyGxhS/I/1moVq9l+8j/o1Db3FdfyP6wmSmal2/I/r9JK70848z/GI8lTnpb1P0YrKVfUWfg/qjFPquF5+z8U/+mvNbj7P/eVGv0tpPw/m8y4rBq8/D+niUxvWjsAQOLcQM8ujQFALcFubcg/BkA=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"yRrgECw38z/epBWjQU7zPydn5ETlCvU/xyE/BtWT9T+OrdJH1Q72P7+t+AOTafY/1mET1Ygp9z9yEAat6Cn3P0EBPmGg+fc/hk4AKTWF+D/A5ZUXKpT4P5PT/A9Jxvg/KSa1RTXT+D98YMxaKEL5P7KRnPSeTPk/tFnsa0RU+T/k1yGHFTz6P+i94LQCh/o/BgGhqFLH+j8h4gOSQNL6P8LJTg5aX/s/5dxfDptf+z/Yt9KKhzX8P/n60qTnc/w/aPzkj7SM/D8WnCLM06b8P2+ZFoxa1fw/i0IWOYXo/D8C/IDoRjr9P3QygisSgv0/m0EmuOeF/T+Has27f7H9P8h6aw9Fsv0/cerARwfE/T9I1n1rxdv9P5Ce7QxL6/0/OXulhTz9/T/aDAeQ7RD+PxIxIQiRNP4/yYijI1ZD/j+7qtWcZV3+P0Z84qz9i/4/sCmKtN7Q/j99HmNK9tj+P5FM37W3Af8/ak8e0SED/z/2bZ+Ehjj/P/UHavega/8/hsHEanjC/z8nRguFOOP/P+hv2Sh/8v8//nE6dpcFAECezgcl3EsAQFUDO8qOoABAYtFSzqegAECO++Oj/cgAQH5FaG3J7ABAeIUSnePuAEBEvJIIPgYBQKk5TotEFgFANZHVw/khAUAe9FrxZksBQMuOmcKBWQFAdM4tM2NaAUDu8wtO/pIBQMgXyTjlqAFAoLlEskrTAUCtLssJQtcBQByLSnyzBwJAhoUZCH0gAkBm/2AvekICQPDPNB9AZgJA9YVAuU+HAkBFcGmgAZkCQFRH7iuinQJANJNuUICmAkDbHjxEihQDQCx+sOUOLQNA+jE0cvVbA0AJDa0KQWMDQJ0c1ChzbQNAnSQiOa3mA0CqdGydSuoDQPRxhgn3KQRA92eWjRcxBECP/qET4GwEQJwgMlvYkgRAtlrFarafBEA6tc19xbUEQKuJklnptgRArLTS+xPOBEDySPKUp2UFQNJKyhV1FgZAasyTanjeBkDFf/prDe4GQH6lRn8LKQdAJzMuqwYvB0DURKY3rR0IQHFuoGeXxghAlmC3NuQfC0A=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"5292"},"selection_policy":{"id":"5293"}},"id":"5277","type":"ColumnDataSource"},{"attributes":{},"id":"5293","type":"UnionRenderers"},{"attributes":{},"id":"5258","type":"UndoTool"},{"attributes":{"overlay":{"id":"5262"}},"id":"5257","type":"LassoSelectTool"},{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5273","type":"Patch"},{"attributes":{},"id":"5253","type":"ResetTool"},{"attributes":{},"id":"5237","type":"DataRange1d"},{"attributes":{},"id":"5254","type":"PanTool"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"E/KbEeId0T/bqsqYMtfRP5PsZjkfkdI/O7dw86dL0z/SCujGzAbUP1nnzLONwtQ/0Ewfuup+1T82O9/Z4zvWP4uyDBN5+dY/0LKnZaq31z8GPLDRd3bYPypOJlfhNdk/PukJ9ub12T9CDVuuiLbaPza6GYDGd9s/GfBFa6A53D/srt9vFvzcP6725o0ov90/YMdbxdaC3j8CIT4WIUffP8oBR8ADBuA/irclAsVo4D/CMbtQ1MvgP3JwB6wxL+E/mnMKFN2S4T86O8SI1vbhP1LHNAoeW+I/UxlcmLO/4j+Cl+UVlDfjPwpTITZ6ueM/KvBPhhg+5D87bAGFpKnkP8iZLyrvK+U/1QkyigiZ5T9TacqFlAnmP8ISnCT8aeY/uYsw+Q7S5j/8HFrX7yTnPyAm4zfcbec/lQyAroHJ5z+Z/Tup0RHoPydfrMb7Yug/5GeA3pq86D+aIsa6qx/pP6yGNa6wjOk/B9+4xTH56T9A9qYCOTjqP5Ds7b9rjeo/5k44iM/y6j9ruGr+Z2PrP7pUyuDsv+s/Jhzk2YkR7D+h6vIP62LsP5CNv4Pfruw/FEuGew8K7T+xYPhsyILtP8LffqDmDu4/k4nAURBr7j94vQ6Lv9DuP/XN9KSWSO8/iJ0xCSuQ7z+41L138OXvP9IjeUbkHfA/BKXfcwAx8D9SKe3hoUPwP8uLf7kJWPA/Az01Oi9v8D9+I+TcHZbwP8CvOvVfyfA/eoFepTf88D/cFSomtyjxP1KqCWkwavE/OiZYc4eg8T8Xyyjb487xP0/FjOfg/vE/eEgH4BQW8j/BbtE0kTTyPxZOc9+xV/I/tgigAMd/8j9Z8GdbKK3yP2yw9DJJ3fI/fsALC6kC8z/2jZZKlSfzPynNqfJXSPM/GbXo6V5g8z9ktUibA5rzPzbhVoAW1fM/eQen5lz68z/+8dtQ3R/0P7IEMGzoOvQ/HLB64Jtf9D9QSucleYf0P0KOJRyBsPQ/35rkyuTU9D+wtBqXVO/0P2Gt7Pf7IPU/uv0+J3Ra9T/AWCC91ob1P3LVQjtHtvU/N6RGFYu69T8rs80dZMr1P9fs50K99PU/RUTnUx8V9j+8PeBzMCr2P2Ro7TO8MvY/B8eIhmQ99j+wonv1XEv2P6fjpoIYXvY/cuJCA2J/9j9iSG4ShKf2P1p1nfxKz/Y/UshZFxnt9j+3UJ+WHyL3PwQZKX85Uvc/ZsgKq2yD9z/VZZF1RLP3P9voqEbS3/c/SkmKA6YR+D/L8atylUn4P1MKa2sGh/g/EahQ5tTJ+D+6Iio1KhP5P4CXYmX4Vfk/QfU9PQmT+T+soCt3wNn5PwyGGYNLLPo/cehXJW1h+j/A2L93VI/6Pw3CO0pxzvo/NVTtxIIM+z/WETbBi1f7P+jn732Zn/s/xIHuJvHm+z8S2qdxZC38P/xDK5vBcvw/P2shaNO2/D8Sc+mT3fj8P3XuYsX3O/0/FKY2VHSK/T+eH20uFeL9Pz2Dqyy+E/4//FolAsFI/j8azLQK8YH+P+lwsRAtwP4/zljwTF8E/z+J8dOFbUT/P1y8SXmCd/8/lRYiDm2s/z//fH/greb/P1VNDhYvEABA3m1+5AgmAEDpB1U/7jQAQFLKY2T1RQBAYZdsJN9OAEC3lbNOcWcAQG7wR/SxeQBAAi2fhCGLAEBFJOiL6JwAQKIvvCU9rwBANjK9cljCAEDLmJWYdtYAQNhZ+MHW6wBAq+yic5YCAUAKcs119xgBQLh+U7DjLgFAumzdwkREAUBcW+AEBFkBQC8vnoUKbQFABZIlDEGAAUD48lEXkJIBQD+66CwFpAFAwM58AxK1AUBWjVCPu8UBQA7BDP/K1QFApBvFI2PlAUC6m3n9g/QBQFBBKowtAwJAZgzXz18RAkD8/H/IGh8CQBITJXZeLAJAp07G2Co5AkC9r2Pwf0UCQFM2/bxdUQJAaOKSPsRcAkD+syR1s2cCQBOrsmArcgJAqcc8ASx8AkC+CcNWtYUCQFRxRWHHjgJAaf7DIGKXAkD+sD6VhZ8CQBSJtb4xpwJAqIYonWauAkC+qZcwJLUCQFPyAnlquwJAaGBqdjnBAkD8880okcYCQBGtLZBxywJApouJrNrPAkC7j+F9zNMCQOKh3V40gxBAm3mOXGuBEECnn9u3cn8QQAUUxXBKfRBAttZKh/J6EEC652z7angQQBBHK82zdRBAuvSF/MxyEEC18HyJtm8QQAQ7EHRwbBBApdM/vPpoEECYugtiVWUQQN/vc2WAYRBAeHN4xntdEEBkRRmFR1kQQKJlVqHjVBBAM9QvG1BQEEAWkaXyjEsQQE2ctyeaRhBA1fVlundBEECxnbCqJTwQQN+Tl/ijNhBAYNgapPIwEEA0azqtESsQQFpM9hMBJRBA0ntO2MAeEECe+UL6UBgQQPrG03mxERBAEvixm4MGEEBDazcH4PUPQKO2dc823g9Ah8Ot7jbID0BmWbJ/UbQPQGzKD7tUog9AbozU/BCSD0DcOJHEWIMPQMyMWLUAdg9A8mi/ld9pD0Cm0dxPzl4PQF637brpVQ9AT1dtU+hOD0DGbutsaUkPQAKzVv8QRQ9ANdH8podBD0Caboqkej4PQOx0gM3vOg9Ayu9gZvIrD0DOaNbonyUPQOCh5Mr3HQ9AMXbRFAEPD0D+SJDe5gEPQOjkaYxC9w5AEzSGcy3xDkBwOleduOUOQNojCZE61A5AdWVyG8m/DkDsPs7f1qsOQOzNyhLClg5Aqov5t2SADkA5qUHHmGgOQIYP4Cw4Tw5AJdosVVIwDkCKxbn7tQ0OQMnfKxHb6g1AjBcOHzDLDUD11+ST2KsNQNOmTXtJig1APUWSBLRmDUCwzwh2SEENQAq+Ey02Gg1AOffJ3t/yDEB2ouS7vcgMQLLqeg54pAxAMrkd50CBDEA99NH0plYMQKgyGuYCHwxANUKZ81XtC0CJZI42wsELQItnBP5qmQtA7ux1oipkC0D+yziy8zkLQMFPRY25FQtAUHIcHE30CkC4yP1lfNcKQIPo/+pgvwpAwFlU8/6oCkB7GDtayJoKQPQACYxviApAHTSsufZtCkDmGWX1+l0KQGAvl0x4TQpAY0xcJ1g7CkAgN7Ou7iYKQLKKVWxjGApAriyd2AwICkCI1NtcuvUJQH1VDZ1M4QlAxSSfVvzJCUDYZXcOu68JQJa50f11kglASiW8tp6ACUAAaVJvWnUJQEpVektVXglA6p2JYrRFCUCZk+odVzMJQKCK+CiMGQlA/gkcO0cHCUB0VisvXOoIQFLpaHdbzwhANFL+9y2+CECc2IZRBqoIQMh9WEt8kAhAeTC1HAZ2CEBu/sVyCmIIQAToWSLFSQhA1jg//FYyCEAv5Ph7kRwIQNvZahp2CAhA1g5EMtD2B0CICvAqD+MHQIa29BGczQdA/1AKK+2vB0C0V4XqkpUHQOWmXltZfgdAvi+CXBBqB0Cm8vrbGVcHQK3i03BiQwdAInMwIOcsB0AKHEdQlRoHQEKc5RmhAgdA8fW2MZX4BkARAX+h6uwGQHfdzVFP3wZA+QErqyTQBkDcHaozq9EGQK9hdCZXzQZAevILsEDGBkD01aDiCb8GQMUn/tq4rQZASwCmUgmbBkCrGOtpFYoGQMLfcDJ8bwZAICtVgVVTBkDvRoCZjkIGQIMUh4AuKgZAY6tPN08TBkBONQ+jAP4FQFCbL6zw6wVAzUTOjrDUBUBnnviBrb0FQNqB2Mi3nQVAqIwHQveBBUBca/Rr7WoFQCuK6X+EWAVAl87F4xdKBUA0A1/pMCwFQEtTgNxDDwVAZpYMyWb1BEBOwgeaIt0EQMq/DXZXxQRAHCNdRtOzBECKBJCYM5cEQKUKF1bPewRAsx0miENeBEC+anvYLTgEQIkAxDBrFARAb3YSAC7zA0CqD9bbUdIDQHDFt1aHqQNAw2gBJxODA0CzhhrTSFoDQDYhC9kTMwNAkV12/PsNA0D7zE6ltfECQDOKJYrQ1QJAwZP6qky6AkCk6c0HKp8CQN6Ln6BohAJAbnpvdQhqAkBUtT2GCVACQJA8CtNrNgJAIhDVWy8dAkAJMJ4gVAQCQEecZSHa6wFA21QrXsHTAUDEWe/WCbwBQASrsYuzpAFAmkhyfL6NAUCGMjGpKncBQMdo7hH4YAFAX+uptiZLAUBNumOXtjUBQJDVG7SnIAFAKj3SDPoLAUAa8YahrfcAQF/xOXLC4wBA+z3rfjjQAEDt1prHD70AQDS8SExIqgBA0u30DOKXAEA=","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"5290"},"selection_policy":{"id":"5291"}},"id":"5272","type":"ColumnDataSource"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"5253"},{"id":"5254"},{"id":"5255"},{"id":"5256"},{"id":"5257"},{"id":"5258"},{"id":"5259"},{"id":"5260"}]},"id":"5263","type":"Toolbar"},{"attributes":{},"id":"5243","type":"LinearScale"},{"attributes":{"data_source":{"id":"5277"},"glyph":{"id":"5278"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5279"},"selection_glyph":null,"view":{"id":"5281"}},"id":"5280","type":"GlyphRenderer"},{"attributes":{},"id":"5290","type":"Selection"},{"attributes":{},"id":"5241","type":"LinearScale"},{"attributes":{"below":[{"id":"5245"}],"center":[{"id":"5248"},{"id":"5252"}],"left":[{"id":"5249"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"5275"},{"id":"5280"}],"title":{"id":"5283"},"toolbar":{"id":"5263"},"toolbar_location":"above","x_range":{"id":"5237"},"x_scale":{"id":"5241"},"y_range":{"id":"5239"},"y_scale":{"id":"5243"}},"id":"5236","subtype":"Figure","type":"Plot"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5274","type":"Patch"},{"attributes":{},"id":"5239","type":"DataRange1d"},{"attributes":{"text":""},"id":"5283","type":"Title"},{"attributes":{"formatter":{"id":"5284"},"ticker":{"id":"5246"}},"id":"5245","type":"LinearAxis"},{"attributes":{},"id":"5246","type":"BasicTicker"},{"attributes":{},"id":"5291","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"5272"},"glyph":{"id":"5273"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5274"},"selection_glyph":null,"view":{"id":"5276"}},"id":"5275","type":"GlyphRenderer"},{"attributes":{"callback":null},"id":"5260","type":"HoverTool"},{"attributes":{"axis":{"id":"5245"},"ticker":null},"id":"5248","type":"Grid"}],"root_ids":["5236"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"d1fa4840-9af4-4734-923e-37a6222d9e0b","root_ids":["5236"],"roots":{"5236":"83b48f8d-ac61-48a6-9b87-4ffc415427b6"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();