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
    
      
      
    
      var element = document.getElementById("277721b5-ed40-430e-abcc-bad46a8d1864");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '277721b5-ed40-430e-abcc-bad46a8d1864' but no matching script tag was found.")
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
                    
                  var docs_json = '{"bc43b2b7-b401-48ca-aff1-45f3df5ff47f":{"roots":{"references":[{"attributes":{"data_source":{"id":"5364"},"glyph":{"id":"5365"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5366"},"selection_glyph":null,"view":{"id":"5368"}},"id":"5367","type":"GlyphRenderer"},{"attributes":{},"id":"5348","type":"WheelZoomTool"},{"attributes":{"axis":{"id":"5337"},"ticker":null},"id":"5340","type":"Grid"},{"attributes":{},"id":"5346","type":"PanTool"},{"attributes":{},"id":"5329","type":"DataRange1d"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"5381"},"selection_policy":{"id":"5380"}},"id":"5364","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"5337"}],"center":[{"id":"5340"},{"id":"5344"}],"left":[{"id":"5341"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"5367"},{"id":"5372"}],"title":{"id":"5375"},"toolbar":{"id":"5355"},"toolbar_location":"above","x_range":{"id":"5329"},"x_scale":{"id":"5333"},"y_range":{"id":"5331"},"y_scale":{"id":"5335"}},"id":"5328","subtype":"Figure","type":"Plot"},{"attributes":{"source":{"id":"5364"}},"id":"5368","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5371","type":"Line"},{"attributes":{},"id":"5383","type":"Selection"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5366","type":"Patch"},{"attributes":{},"id":"5381","type":"Selection"},{"attributes":{},"id":"5382","type":"UnionRenderers"},{"attributes":{},"id":"5331","type":"DataRange1d"},{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5365","type":"Patch"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5370","type":"Line"},{"attributes":{},"id":"5378","type":"BasicTickFormatter"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"5345"},{"id":"5346"},{"id":"5347"},{"id":"5348"},{"id":"5349"},{"id":"5350"},{"id":"5351"},{"id":"5352"}]},"id":"5355","type":"Toolbar"},{"attributes":{},"id":"5351","type":"SaveTool"},{"attributes":{"source":{"id":"5369"}},"id":"5373","type":"CDSView"},{"attributes":{"text":""},"id":"5375","type":"Title"},{"attributes":{},"id":"5380","type":"UnionRenderers"},{"attributes":{},"id":"5345","type":"ResetTool"},{"attributes":{},"id":"5376","type":"BasicTickFormatter"},{"attributes":{"axis":{"id":"5341"},"dimension":1,"ticker":null},"id":"5344","type":"Grid"},{"attributes":{"formatter":{"id":"5378"},"ticker":{"id":"5338"}},"id":"5337","type":"LinearAxis"},{"attributes":{"callback":null},"id":"5352","type":"HoverTool"},{"attributes":{},"id":"5333","type":"LinearScale"},{"attributes":{"overlay":{"id":"5353"}},"id":"5347","type":"BoxZoomTool"},{"attributes":{},"id":"5338","type":"BasicTicker"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"5354","type":"PolyAnnotation"},{"attributes":{},"id":"5342","type":"BasicTicker"},{"attributes":{"formatter":{"id":"5376"},"ticker":{"id":"5342"}},"id":"5341","type":"LinearAxis"},{"attributes":{},"id":"5335","type":"LinearScale"},{"attributes":{"data":{"x":{"__ndarray__":"HF+T8x7xCMCQswWh1XkFwFM5i5fDCQTAtwbY7lbOAMAjvLa/5ib/v1U/6tv8OPy//I36TvEZ+7/C3/eyRGH6v3zabwrP6fm/UWUoJ3CN+L8Gu+M40rH2v4Siej5RgfW/6Q1oSMdp9L9+RUwUxjLyv9k7SCA8l/G/nw4c2zdl8L/mOfk0lVnwvwCOXv7Zde+/90cVo2b+6L+0CSurIqrnv3Z+5FBYKee/4A/Bw/Ho5L/bD7S9SJTjv3sc5wWng+O/FpFeyNbt4r9rPdcVMjjivwcnNIooqOG/6AbMRsE+4L/+A6uwtMjdvz1RZhwkdtu/HTlTg8L60L+e6MNysK7MvzWXw2++Ycm/6VBwJ4SjyL/V+jnJkIfAvzoR78tHbcC/QYcOAkyMvb/VPPNPXiy6vzPKJ1QZX7S/xJ6iIgfElz8oMVuH7wifP69j/15QmaA/6AOe/SdQsD82yhz7qke+Py0PLhyv9cM/Fa+DT2eoyj+PLF3jh+DLPzOpqZCTC8w/G0E5F9UAzz/WqQLbSXbQP/e3YRTeXdM/uwzylfC52D/DztHV8xnZPyUW+aaPfdk/IocxT2aG2T9h5CgWJ5HZP7V8lus3kNo/gqmB10KR2j8439jz2yTcP+Hn/wAMR94/r1y13eEL3z/PgHmpDw/gP6K0++wwWOA/WQmBi4+O4D8gtWkiMlvjP34AlZzshuU/CIk+yfUM5j8J8z+TsWzoP0wsCjhXMOo/1m4CRY0U7D+RzDBG3CPsP9DUyfP5J+0/DIldpvdU7T8/IaU4/ODtP850180HWe8/NQaDjB6B8D+1pv8TVczwPwj9tiIN6fA/UwUIru718D8HsGYKTvvwP6pnn6r3bvE/KF48T7Oi8T/2ptLrJRXyP222zgI/MvM/0nofjwXs9j8PUa1ZDQX3P5vrfJP6y/c/BU4HtQRC+D+yqeK5G7r5P/VTDuunVvo/C2ldu7i4+j8yjp0iiB37PydiYm0NVfw/nRK15iP7/D/qOtNl8vz+PyBjANl8OP8/5JH5doK7AEDgvOj5RYwGQElGJEvv5gZA6Y3EbdkxCkA=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"kIOyMYQ73D/gmPS9VAzlP1qN6dB47Oc/kvJPIlJj7j/uoSSgjGzwP1bgCpKB4/E/ArmCWAdz8j8fEISmXc/yP8ISyHoYC/M/WM1r7Ee58z99Io7jFqf0P76uwmBXP/U/DPnLWxzL9T9B3dn1nOb2PxTi2+9hNPc/sPhxEmTN9z8NY4NlNdP3P4BcaICJIvg/Aq46V2bA+T+TPTVVdxX6P2LgxuupNfo/CLwPj8PF+j8J/JLQ7Rr7P+E4hj4WH/s/ulvoTYpE+z+lMIp683H7Pz72ct31lfs/Rv5Mrk/w+z+An+pp6Ub8P9g1c3w7kfw/3JiVr6fg/T92wdP4FDX+P43GAxnkaf4/8fqIvcd1/j9TYGzzhvf+P+wOQYMr+f4/xovvn50T/z8ZZoANnS7/P67BXjUHXf8/n6IiB8QXAEAxW4fvCB8AQMf+vaAyIQBAEHj2n0BBAEApc+yrHnkAQHlw4XitnwBAeR18OkPVAEBk6Ro/BN8AQEpNhZxc4ABACcq5qAb4AECdKrCdZAcBQH8bRuHdNQFAzCBfCZ+LAUDsHF09n5EBQGKRb/rYlwFAchjzZGaYAUBGjmJxEpkBQMtnuX4DqQFAmBp4LRSpAUD0jT2/TcIBQH7+D8Bw5AFAy1XbHb7wAUAaMC/14QECQJR2nx0GCwJAKyFw8dERAkCkNk1EZmsCQBCgkpPdsAJAIdEnuZ7BAkBh/mcylg0DQIpFAecKRgNA202gqJGCA0CSGcaIe4QDQJo6eT7/pANAIrHL9J6qA0AopBSHH7wDQJruuvkg6wNAjcEgo0cgBECt6f9EFTMEQEK/rUhDOgRAVQGCq3s9BEACrJmC0z4EQOrZp+q9WwRAihfP06xoBEC+qfR6SYUEQJuts8CPzARAtN7HYwG7BUBEVGtWQ8EFQOc636T+8gVAgdNBLYEQBkBsqnjuhm4GQP2Uw/qplQZAQ1rXLi6uBkCMY6cIYscGQIqYWFtDFQdAp0St+cg+B0C6znSZPL8HQMgYQDYfzgdA8sh8O8FdCEBwXvT8IkYLQCQjkqV3cwtA9EbituwYDUA=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"5383"},"selection_policy":{"id":"5382"}},"id":"5369","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"5369"},"glyph":{"id":"5370"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5371"},"selection_glyph":null,"view":{"id":"5373"}},"id":"5372","type":"GlyphRenderer"},{"attributes":{},"id":"5350","type":"UndoTool"},{"attributes":{"overlay":{"id":"5354"}},"id":"5349","type":"LassoSelectTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"5353","type":"BoxAnnotation"}],"root_ids":["5328"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"bc43b2b7-b401-48ca-aff1-45f3df5ff47f","root_ids":["5328"],"roots":{"5328":"277721b5-ed40-430e-abcc-bad46a8d1864"}}];
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